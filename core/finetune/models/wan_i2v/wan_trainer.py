# Modified from: https://github.com/huggingface/finetrainers
# Wan I2V trainer base class adapted from the finetrainers project.

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from diffusers import (
    AutoencoderKLWan,
    UniPCMultistepScheduler,
    WanImageToVideoPipeline,
    WanTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
import numpy as np
from numpy import dtype
from transformers import T5TokenizerFast, UMT5EncoderModel
from typing_extensions import override

from core.finetune.schemas import Wan_Components as Components
from core.finetune.trainer import Trainer
from core.finetune.utils import unwrap_model
from ..utils import register
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.optimization import get_scheduler
from diffusers.configuration_utils import register_to_config
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.modeling_utils import ModelMixin
import json
from core.finetune.schemas import Args, Components, State
from accelerate.accelerator import Accelerator, DistributedType
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
import types
from termcolor import cprint
from core.finetune.utils import (
    cast_training_params,
    free_memory,
    get_intermediate_ckpt_path,
    get_latest_ckpt_path_to_resume_from,
    get_memory_statistics,
    get_optimizer,
    string_to_filename,
    unload_model,
    unwrap_model,
)
from tqdm import tqdm
from itertools import chain
import regex as re
import ftfy
import html

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

try:
    from diffusers.models.transformers.transformer_wan import WanTimeTextImageEmbedding
except ImportError as e:
    cprint("❌ Critical Error: Could not import `WanTimeTextImageEmbedding` for monkey-patching.", 'red')
    cprint("   The structure of the `diffusers` library may have changed.", 'red')
    raise e

def patched_wan_time_text_image_embedding_forward(
    self,  # The first argument must be `self`
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    timestep_seq_len: Optional[int] = None,
) -> tuple:
    timestep = self.timesteps_proj(timestep.to(torch.float32))
    if timestep_seq_len is not None:
        timestep = timestep.unflatten(0, (-1, timestep_seq_len))

    with torch.autocast(device_type=timestep.device.type, dtype=torch.float32, enabled=True):
        temb = self.time_embedder(timestep)

        timestep_proj = self.time_proj(self.act_fn(temb))
    temb_casted = temb.type_as(encoder_hidden_states)
    timestep_proj = timestep_proj.type_as(encoder_hidden_states)
    
    encoder_hidden_states = self.text_embedder(encoder_hidden_states)

    if encoder_hidden_states_image is not None:
        encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

    return temb_casted, timestep_proj, encoder_hidden_states, encoder_hidden_states_image

WanTimeTextImageEmbedding.forward = patched_wan_time_text_image_embedding_forward
cprint("✅ [Monkey Patch Applied] `WanTimeTextImageEmbedding.forward` has been replaced to ensure float32 stability during mixed-precision training.", "green")


class Wan_Components(Components):
    high_noise_model : Any = None
    low_noise_model : Any = None

class WanI2VTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder"]
    @override
    def __init__(self, args: Args) -> None:
        self.args = args
        self.state = State(
            weight_dtype=self.get_training_dtype(),
            train_frames=self.args.train_resolution[0],
            train_height=self.args.train_resolution[1],
            train_width=self.args.train_resolution[2],
        )

        self.components: Components = self.load_components()
        self.accelerator: Accelerator = None
        self.dataset: Dataset = None
        self.data_loader: DataLoader = None

        self.optimizer = None
        self.lr_scheduler = None

        self._init_distributed()
        self._init_logging()
        self._init_directories()

        self.state.using_deepspeed = self.accelerator.state.deepspeed_plugin is not None

    def get_training_dtype(self) -> torch.dtype:
        _DTYPE_MAP = {
            "fp32": torch.float32,
            "fp16": torch.float16,  # FP16 is Only Support for CogVideoX-2B
            "bf16": torch.bfloat16,
        }
        if self.args.mixed_precision == "no":
            return _DTYPE_MAP["fp32"]
        elif self.args.mixed_precision == "fp16":
            return _DTYPE_MAP["fp16"]
        elif self.args.mixed_precision == "bf16":
            return _DTYPE_MAP["bf16"]
        else:
            raise ValueError(f"Invalid mixed precision: {self.args.mixed_precision}")

    @override
    def __prepare_saving_loading_hooks(self, transformer_lora_config):
        def save_model_hook(models: list, weights: list, output_dir: str):
            if self.accelerator.is_main_process:
                unwrapped_high_noise_model = unwrap_model(self.accelerator, self.components.high_noise_model)
                unwrapped_low_noise_model = unwrap_model(self.accelerator, self.components.low_noise_model)
                
                high_noise_lora_layers_to_save = get_peft_model_state_dict(
                    unwrapped_high_noise_model, adapter_name="high_noise"
                )
                low_noise_lora_layers_to_save = get_peft_model_state_dict(
                    unwrapped_low_noise_model, adapter_name="low_noise"
                )

                self.components.pipeline_cls.save_lora_weights(
                    save_directory=os.path.join(output_dir, "high_noise_lora"),
                    transformer_lora_layers=high_noise_lora_layers_to_save,
                )
                
                self.components.pipeline_cls.save_lora_weights(
                    save_directory=os.path.join(output_dir, "low_noise_lora"),
                    transformer_lora_layers=low_noise_lora_layers_to_save,
                )
                
                logger.info(f"Successfully saved high-noise and low-noise LoRA weights to {output_dir}")

                indices_to_pop = []
                for i, model in enumerate(models):
                    if model is self.components.high_noise_model or model is self.components.low_noise_model:
                        indices_to_pop.append(i)

                for i in sorted(indices_to_pop, reverse=True):
                    weights.pop(i)
                    models.pop(i)

        def load_model_hook(models: list, input_dir: str):
            high_noise_model_ = unwrap_model(self.accelerator, self.components.high_noise_model)
            low_noise_model_ = unwrap_model(self.accelerator, self.components.low_noise_model)

            high_noise_lora_path = os.path.join(input_dir, "high_noise_lora")
            low_noise_lora_path = os.path.join(input_dir, "low_noise_lora")
            
            if os.path.exists(high_noise_lora_path):
                high_noise_state_dict = self.components.pipeline_cls.lora_state_dict(high_noise_lora_path)
                set_peft_model_state_dict(high_noise_model_, high_noise_state_dict, adapter_name="high_noise")
                logger.info(f"Successfully loaded LoRA weights from {high_noise_lora_path} into high_noise_model")
            else:
                logger.warning(f"Could not find LoRA weights for high_noise_model at {high_noise_lora_path}")

            if os.path.exists(low_noise_lora_path):
                low_noise_state_dict = self.components.pipeline_cls.lora_state_dict(low_noise_lora_path)
                set_peft_model_state_dict(low_noise_model_, low_noise_state_dict, adapter_name="low_noise")
                logger.info(f"Successfully loaded LoRA weights from {low_noise_lora_path} into low_noise_model")
            else:
                logger.warning(f"Could not find LoRA weights for low_noise_model at {low_noise_lora_path}")
            
            indices_to_pop = []
            for i, model in enumerate(models):
                if model is self.components.high_noise_model or model is self.components.low_noise_model:
                    indices_to_pop.append(i)
            
            for i in sorted(indices_to_pop, reverse=True):
                models.pop(i)

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    @override
    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")
        weight_dtype = self.state.weight_dtype

        if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError("Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead.")

        # For LoRA, we freeze all the parameters
        # For SFT, we train all the parameters in transformer model
        for attr_name, component in vars(self.components).items():
            if hasattr(component, "requires_grad_"):
                if self.args.training_type == "sft" and ("high_noise_model" in attr_name or "low_noise_model" in attr_name):
                    component.requires_grad_(True)
                else:
                    component.requires_grad_(False)

        if self.args.training_type == "lora":
            transformer_lora_config = LoraConfig(
                r=self.args.rank,
                lora_alpha=self.args.lora_alpha,
                init_lora_weights=True,
                target_modules=self.args.target_modules,
            )
            # self.components.high_noise_model.add_adapter(transformer_lora_config)
            # self.components.low_noise_model.add_adapter(transformer_lora_config)
            self.components.high_noise_model.add_adapter(transformer_lora_config, adapter_name="high_noise")
            self.components.low_noise_model.add_adapter(transformer_lora_config, adapter_name="low_noise")
            self.components.high_noise_model.requires_grad_(False)
            self.components.low_noise_model.requires_grad_(False)
            for name, param in self.components.high_noise_model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True
            for name, param in self.components.low_noise_model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True
            self.__prepare_saving_loading_hooks(transformer_lora_config)

        # Load components needed for training to GPU (except transformer), and cast them to the specified data type
        ignore_list = ["high_noise_model", "low_noise_model"] + self.UNLOAD_LIST
        self.move_components_to_device(dtype=weight_dtype, ignore_list=ignore_list)

        if self.args.gradient_checkpointing:
            self.components.high_noise_model.enable_gradient_checkpointing()
            self.components.low_noise_model.enable_gradient_checkpointing()
            cprint("✅ Gradient checkpointing enabled for both transformer models.", "green")

    @override
    def prepare_optimizer(self) -> None:
        logger.info("Initializing optimizer and lr scheduler")

        # Make sure the trainable params are in float32
        cast_training_params([self.components.high_noise_model, self.components.low_noise_model], dtype=torch.float32)
        
        # For LoRA, we only want to train the LoRA weights
        # For SFT, we want to train all the 
        trainable_parameters_high = list(filter(lambda p: p.requires_grad, self.components.high_noise_model.parameters()))
        trainable_parameters_low = list(filter(lambda p: p.requires_grad, self.components.low_noise_model.parameters()))
        trainable_parameters = trainable_parameters_high + trainable_parameters_low
        transformer_parameters_with_lr = {
            "params": trainable_parameters,
            "lr": self.args.learning_rate,
        }
        params_to_optimize = [transformer_parameters_with_lr]
        self.state.num_trainable_parameters = sum(p.numel() for p in trainable_parameters)

        use_deepspeed_opt = (
            self.accelerator.state.deepspeed_plugin is not None
            and "optimizer" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.learning_rate,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_deepspeed=use_deepspeed_opt,
        )

        num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        if self.args.train_steps is None:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True

        use_deepspeed_lr_scheduler = (
            self.accelerator.state.deepspeed_plugin is not None
            and "scheduler" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        total_training_steps = self.args.train_steps * self.accelerator.num_processes
        num_warmup_steps = self.args.lr_warmup_steps * self.accelerator.num_processes

        if use_deepspeed_lr_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                total_num_steps=total_training_steps,
                num_warmup_steps=num_warmup_steps,
            )
        else:
            lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_training_steps,
                num_cycles=self.args.lr_num_cycles,
                power=self.args.lr_power,
            )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    @override
    def prepare_for_training(self) -> None:
        high_noise_model, low_noise_model, self.optimizer, self.data_loader, self.lr_scheduler = self.accelerator.prepare(
            self.components.high_noise_model, 
            self.components.low_noise_model, 
            self.optimizer, 
            self.data_loader, 
            self.lr_scheduler
        )
        self.components.high_noise_model = high_noise_model
        self.components.low_noise_model = low_noise_model

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        if self.state.overwrote_max_train_steps:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.args.train_epochs = math.ceil(self.args.train_steps / num_update_steps_per_epoch)
        self.state.num_update_steps_per_epoch = num_update_steps_per_epoch

    @override
    def load_components(self) -> Dict[str, Any]:
        components = Wan_Components()
        model_path = str(self.args.model_path)

        cprint(f"Loading components from: {model_path}",'green')
        components.pipeline_cls = WanImageToVideoPipeline
        components.tokenizer = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer")
        components.text_encoder = UMT5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
        # components.transformer = WanTransformer3DModelDembSameRope.from_pretrained(model_path, subfolder="transformer")
        components.vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae")
        components.scheduler = UniPCMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
        components.high_noise_model = WanTransformer3DModel.from_pretrained(model_path, subfolder="transformer",eps=1e-5)
        components.low_noise_model = WanTransformer3DModel.from_pretrained(model_path, subfolder="transformer_2",eps=1e-5)
        components.transformer = nn.ModuleList([
            components.high_noise_model,
            components.low_noise_model
        ])

        # with open(os.path.join(model_path, "model_index.json"), "r") as f:
        #     model_index = json.load(f)
        #     boundary_ratio = model_index.get("boundary_ratio", 0.9)
        # boundary_ratio = components.pipeline_cls.config.boundary_ratio
        boundary_ratio = 0.9
        # self.state.moe_boundary = self.args.moe_boundary or 500
        # logger.info(f"MoE boundary set to timestep {self.state.moe_boundary}")
        num_train_timesteps = components.scheduler.config.num_train_timesteps
        self.state.moe_boundary = int(num_train_timesteps * boundary_ratio)

        return components

    @override
    def prepare_models(self) -> None:
        self.state.transformer_config = self.components.high_noise_model.config

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W]
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        # latent = latent_dist.sample()
        latent = latent_dist.mode()
        latents_mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(latent.device, latent.dtype)
        )
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
            latent.device, latent.dtype
        )
        latent = (latent - latents_mean) * latents_std
        return latent

    def _get_t5_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        dtype = dtype or self.components.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        def basic_clean(text):
            text = ftfy.fix_text(text)
            text = html.unescape(html.unescape(text))
            return text.strip()

        def whitespace_clean(text):
            text = re.sub(r"\s+", " ", text)
            text = text.strip()
            return text

        def prompt_clean(text):
            text = whitespace_clean(basic_clean(text))
            return text

        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.components.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: str | list[str],
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_embedding = self._get_t5_prompt_embeds(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

        return prompt_embedding

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        with torch.no_grad():
            prompt_token_ids = self.components.tokenizer(
                prompt,
                padding="max_length",
                max_length=512,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            prompt_token_ids = prompt_token_ids.input_ids
            # TODO: should be pass in attention mask?
            prompt_embedding = self.components.text_encoder(prompt_token_ids.to(self.accelerator.device))[0]
        return prompt_embedding

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {"encoded_videos": [], "prompt_embedding": [], "images": [], "null_embedding": []}
        for sample in samples:
            encoded_video = sample["encoded_video"]
            prompt_embedding = sample["prompt_embedding"]
            image = sample["image"]
            null_embedding = sample["null_embedding"]

            ret["encoded_videos"].append(encoded_video)
            ret["prompt_embedding"].append(prompt_embedding)
            ret["images"].append(image)
            ret["null_embedding"].append(null_embedding)

        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
        ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])
        ret["images"] = torch.stack(ret["images"])
        ret["null_embedding"] = torch.stack(ret["null_embedding"])
        return ret

    @override
    def prepare_dataset(self) -> None:
        logger.info("Initializing dataset and dataloader")
        from core.finetune.datasets import I2VDataset
        if self.args.model_type == "wan-i2v":
            self.dataset = I2VDataset(
                data_root=self.args.validation_dir,
                max_num_frames=self.args.train_resolution[0],
                height=self.args.train_resolution[1],
                width=self.args.train_resolution[2],
                device=self.accelerator.device,
                trainer=self  
            )
        else:
            raise ValueError(f"Invalid model type: {self.args.model_type}")

        # Prepare VAE and text encoder for encoding
        self.components.vae.requires_grad_(False)
        self.components.text_encoder.requires_grad_(False)
        self.components.vae = self.components.vae.to(self.accelerator.device, dtype=self.state.weight_dtype)
        self.components.text_encoder = self.components.text_encoder.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )

        # Precompute latent for video and prompt embedding
        logger.info("Precomputing latent for video and prompt embedding ...")
        tmp_data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=1,
            num_workers=0,
            pin_memory=self.args.pin_memory,
        )
        tmp_data_loader = self.accelerator.prepare_data_loader(tmp_data_loader)
        for _ in tmp_data_loader:
            ...
        self.accelerator.wait_for_everyone()
        logger.info("Precomputing latent for video and prompt embedding ... Done")

        unload_model(self.components.vae)
        unload_model(self.components.text_encoder)
        free_memory()

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=True,
        )

    # @override
    # def compute_loss(self, batch) -> torch.Tensor:
    #     PROMPT_DROPOUT_RATE = 0.1
    #     CROSSOVER_PROBABILITY = 0.3

    #     model_dtype = self.components.high_noise_model.dtype
    #     device = self.components.high_noise_model.device
    #     prompt_embedding = batch["prompt_embedding"].to(model_dtype)
    #     latent = batch["encoded_videos"].to(model_dtype)
    #     images = batch["images"]
    #     null_embedding = batch["null_embedding"].to(model_dtype)

    #     batch_size, num_channels, num_frames, height, width = latent.shape
    #     vae_scale_factor_temporal = 2 ** sum(self.components.vae.config.temperal_downsample)
    #     num_real_frames = (num_frames - 1) * vae_scale_factor_temporal + 1
        
    #     images_unsqueezed = images.unsqueeze(2)
    #     video_condition = torch.cat([images_unsqueezed, images_unsqueezed.new_zeros(batch_size, images.shape[1], num_real_frames - 1, images.shape[2], images.shape[3])], dim=2)
    #     with torch.inference_mode():
    #         latent_condition = self.encode_video(video_condition)
    #     mask_lat_size = torch.ones(batch_size, 1, num_real_frames, latent_condition.shape[3], latent_condition.shape[4], device=latent.device)
    #     mask_lat_size[:, :, 1:] = 0
    #     first_frame_mask = mask_lat_size[:, :, 0:1]
    #     # first_frame_mask[:, :, :, :, latent_condition.shape[4]//2:] = 0.5
    #     first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=vae_scale_factor_temporal)
    #     mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
    #     mask_lat_size = mask_lat_size.view(batch_size, -1, vae_scale_factor_temporal, latent_condition.shape[-2], latent_condition.shape[-1])
    #     mask_lat_size = mask_lat_size.transpose(1, 2).contiguous()
    #     mask_lat_size = mask_lat_size.to(latent_condition)

    #     # Check if they match in all dimensions except channel
    #     assert mask_lat_size.shape[2:] == latent_condition.shape[2:], "Time/Space dimensions mismatch!"

    #     y_condition = torch.concat([mask_lat_size, latent_condition], dim=1) 
    #     timesteps = torch.randint(0, self.components.scheduler.config.num_train_timesteps, (batch_size,), device=latent.device)
    #     timesteps = timesteps.long()

    #     noise = torch.randn_like(latent)
    #     noisy_latents = self.components.scheduler.add_noise(latent, noise, timesteps)
    #     target = noise

    #     prompt_dropout_mask = torch.rand(batch_size, device=latent.device) < PROMPT_DROPOUT_RATE
    #     if prompt_dropout_mask.any():
    #         prompt_embedding[prompt_dropout_mask] = null_embedding.to(dtype=prompt_embedding.dtype)
    
    #     model_kwargs = {
    #         'encoder_hidden_states': prompt_embedding,
    #     }
    #     high_noise_model = self.components.high_noise_model
    #     low_noise_model = self.components.low_noise_model
    #     primary_high_noise_mask = (timesteps >= self.state.moe_boundary)
    #     primary_low_noise_mask = (timesteps < self.state.moe_boundary)

    #     crossover_draw = torch.rand(batch_size, device=device)
    #     perform_crossover_mask = (crossover_draw < CROSSOVER_PROBABILITY)
        
    #     high_noise_mask = (primary_high_noise_mask & ~perform_crossover_mask) | (primary_low_noise_mask & perform_crossover_mask)
    #     low_noise_mask = (primary_low_noise_mask & ~perform_crossover_mask) | (primary_high_noise_mask & perform_crossover_mask)

    #     predicted_noise = torch.zeros_like(target)           

    #     if high_noise_mask.any():
    #         high_noise_latents = noisy_latents[high_noise_mask]
    #         high_noise_y = y_condition[high_noise_mask]
    #         high_noise_kwargs = {k: v[high_noise_mask] for k, v in model_kwargs.items() if v is not None}
        
    #         high_noise_input = torch.cat([high_noise_latents, high_noise_y], dim=1)

    #         high_noise_pred = high_noise_model(
    #             high_noise_input,
    #             timestep=timesteps[high_noise_mask],
    #             **high_noise_kwargs,
    #         )[0]
    #         predicted_noise[high_noise_mask] = high_noise_pred.to(predicted_noise.dtype)

    #     if low_noise_mask.any():
    #         low_noise_latents = noisy_latents[low_noise_mask]
    #         low_noise_y = y_condition[low_noise_mask]
    #         low_noise_kwargs = {k: v[low_noise_mask] for k, v in model_kwargs.items() if v is not None}

    #         low_noise_input = torch.cat([low_noise_latents, low_noise_y], dim=1)

    #         low_noise_pred = low_noise_model(
    #             low_noise_input,
    #             timestep=timesteps[low_noise_mask],
    #             **low_noise_kwargs,
    #         )[0]
    #         predicted_noise[low_noise_mask] = low_noise_pred.to(predicted_noise.dtype)

    #     loss = F.mse_loss(predicted_noise.float(), target.float(), reduction="mean")
    #     return loss

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        PROMPT_DROPOUT_RATE = 0.1
        CROSSOVER_PROBABILITY = 0.3

        model_dtype = self.components.high_noise_model.dtype
        device = self.components.high_noise_model.device
        prompt_embedding = batch["prompt_embedding"].to(model_dtype)
        latent = batch["encoded_videos"].to(model_dtype)
        images = batch["images"]
        null_embedding = batch["null_embedding"].to(model_dtype)

        batch_size, num_channels, num_frames, height, width = latent.shape
        vae_scale_factor_temporal = 2 ** sum(self.components.vae.config.temperal_downsample)
        num_real_frames = (num_frames - 1) * vae_scale_factor_temporal + 1
        
        images_unsqueezed = images.unsqueeze(2)
        video_condition = torch.cat([images_unsqueezed, images_unsqueezed.new_zeros(batch_size, images.shape[1], num_real_frames - 1, images.shape[2], images.shape[3])], dim=2)
        with torch.inference_mode():
            latent_condition = self.encode_video(video_condition)
        mask_lat_size = torch.ones(batch_size, 1, num_real_frames, latent_condition.shape[3], latent_condition.shape[4], device=latent.device)
        mask_lat_size[:, :, 1:] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        # first_frame_mask[:, :, :, :, latent_condition.shape[4]//2:] = 0.5
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, vae_scale_factor_temporal, latent_condition.shape[-2], latent_condition.shape[-1])
        mask_lat_size = mask_lat_size.transpose(1, 2).contiguous()
        mask_lat_size = mask_lat_size.to(latent_condition)

        # Check if they match in all dimensions except channel
        assert mask_lat_size.shape[2:] == latent_condition.shape[2:], "Time/Space dimensions mismatch!"

        y_condition = torch.concat([mask_lat_size, latent_condition], dim=1) 
        timesteps = torch.randint(0, self.components.scheduler.config.num_train_timesteps, (batch_size,), device=latent.device)
        timesteps = timesteps.long()

        noise = torch.randn_like(latent)
        noisy_latents = self.components.scheduler.add_noise(latent, noise, timesteps)
        target = noise

        prompt_dropout_mask = torch.rand(batch_size, device=latent.device) < PROMPT_DROPOUT_RATE
        if prompt_dropout_mask.any():
            prompt_embedding[prompt_dropout_mask] = null_embedding.to(dtype=prompt_embedding.dtype)
    
        model_kwargs = {
            'encoder_hidden_states': prompt_embedding,
        }
        high_noise_model = self.components.high_noise_model
        low_noise_model = self.components.low_noise_model
        primary_high_noise_mask = (timesteps >= self.state.moe_boundary)
        primary_low_noise_mask = (timesteps < self.state.moe_boundary)

        crossover_draw = torch.rand(batch_size, device=device)
        perform_crossover_mask = (crossover_draw < CROSSOVER_PROBABILITY)
        
        high_noise_mask = (primary_high_noise_mask & ~perform_crossover_mask) | (primary_low_noise_mask & perform_crossover_mask)
        low_noise_mask = (primary_low_noise_mask & ~perform_crossover_mask) | (primary_high_noise_mask & perform_crossover_mask)

        predicted_noise = torch.zeros_like(target)           

        # if high_noise_mask.any():
        #     high_noise_latents = noisy_latents[high_noise_mask]
        #     high_noise_y = y_condition[high_noise_mask]
        #     high_noise_kwargs = {k: v[high_noise_mask] for k, v in model_kwargs.items() if v is not None}
        
        #     high_noise_input = torch.cat([high_noise_latents, high_noise_y], dim=1)

        #     high_noise_pred = high_noise_model(
        #         high_noise_input,
        #         timestep=timesteps[high_noise_mask],
        #         **high_noise_kwargs,
        #     )[0]
        #     predicted_noise[high_noise_mask] = high_noise_pred.to(predicted_noise.dtype)

        # if low_noise_mask.any():
        #     low_noise_latents = noisy_latents[low_noise_mask]
        #     low_noise_y = y_condition[low_noise_mask]
        #     low_noise_kwargs = {k: v[low_noise_mask] for k, v in model_kwargs.items() if v is not None}

        #     low_noise_input = torch.cat([low_noise_latents, low_noise_y], dim=1)

        #     low_noise_pred = low_noise_model(
        #         low_noise_input,
        #         timestep=timesteps[low_noise_mask],
        #         **low_noise_kwargs,
        #     )[0]
        #     predicted_noise[low_noise_mask] = low_noise_pred.to(predicted_noise.dtype)

        # loss = F.mse_loss(predicted_noise.float(), target.float(), reduction="mean")
 
        high_noise_latents = noisy_latents
        high_noise_y = y_condition
        high_noise_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
        high_noise_input = torch.cat([high_noise_latents, high_noise_y], dim=1)
        print('timesteps',timesteps.shape)
        high_noise_pred = high_noise_model(
            high_noise_input,
            timestep=timesteps,
            **high_noise_kwargs,
        )[0]
        high_noise_predicted_noise = high_noise_pred.to(predicted_noise.dtype)
        high_noise_loss = F.mse_loss(high_noise_predicted_noise.float(), target.float(), reduction="mean")


        low_noise_latents = noisy_latents
        low_noise_y = y_condition
        low_noise_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

        low_noise_input = torch.cat([low_noise_latents, low_noise_y], dim=1)
        low_noise_pred = low_noise_model(
            low_noise_input,
            timestep=timesteps,
            **low_noise_kwargs,
        )[0]
        low_noise_predicted_noise = low_noise_pred.to(predicted_noise.dtype)
        low_noise_loss = F.mse_loss(low_noise_predicted_noise.float(), target.float(), reduction="mean")
        loss = high_noise_loss + low_noise_loss
        return loss


    @override
    def train(self) -> None:
        logger.info("Starting training")

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.total_batch_size_count = (self.args.batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps)
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.dataset),
            "train epochs": self.args.train_epochs,
            "train steps": self.args.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.data_loader),
            "train batch size total count": self.state.total_batch_size_count,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        global_step = 0
        first_epoch = 0
        initial_global_step = 0

        # Potentially load in the weights and states from a previous save
        (
            resume_from_checkpoint_path,
            initial_global_step,
            global_step,
            first_epoch,
        ) = get_latest_ckpt_path_to_resume_from(
            resume_from_checkpoint=self.args.resume_from_checkpoint,
            num_update_steps_per_epoch=self.state.num_update_steps_per_epoch,
            output_dir=self.args.output_dir,
        )
        if resume_from_checkpoint_path is not None:
            self.accelerator.load_state(resume_from_checkpoint_path)

        progress_bar = tqdm(
            range(0, self.args.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.accelerator.is_local_main_process,
        )

        accelerator = self.accelerator
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        free_memory()
        for epoch in range(first_epoch, self.args.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.args.train_epochs})")

            self.components.high_noise_model.train()
            self.components.low_noise_model.train()
            # models_to_accumulate = [self.components.transformer]
            models_to_accumulate = [self.components.high_noise_model, self.components.low_noise_model]

            for step, batch in enumerate(self.data_loader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}

                with accelerator.accumulate(models_to_accumulate):
                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss
                    loss = self.compute_loss(batch)
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            # grad_norm = self.components.transformer.get_global_grad_norm()
                            grad_norm_high = self.components.high_noise_model.get_global_grad_norm()
                            grad_norm_low = self.components.low_noise_model.get_global_grad_norm()
                            grad_norm = (grad_norm_high**2 + grad_norm_low**2)**0.5
                            # In some cases the grad norm may not return a float
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()
                        else:
                            # grad_norm = accelerator.clip_grad_norm_(
                            #     self.components.transformer.parameters(), self.args.max_grad_norm
                            # )
                            # grad_norm = accelerator.clip_grad_norm_(
                            #     self.components.high_noise_model.parameters()+self.components.low_noise_model.parameters(), self.args.max_grad_norm
                            # )
                            grad_norm = accelerator.clip_grad_norm_(
                                chain(self.components.high_noise_model.parameters(), 
                                    self.components.low_noise_model.parameters()), 
                                self.args.max_grad_norm
                            )
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()

                        logs["grad_norm"] = grad_norm

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    self._maybe_save_checkpoint(global_step)

                logs["loss"] = loss.detach().item()
                logs["lr"] = self.lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(logs)

                accelerator.log(logs, step=global_step)

                if global_step >= self.args.train_steps:
                    break

            memory_statistics = get_memory_statistics()
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

        accelerator.wait_for_everyone()
        self._maybe_save_checkpoint(global_step, must_save=True)
        if self.args.do_validation:
            free_memory()
            self.validate(global_step)

        del self.components
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()

    def fit(self):
        self.prepare_models()
        self.prepare_dataset()
        self.prepare_trainable_parameters()
        self.prepare_optimizer()
        self.prepare_for_training()
        self.prepare_trackers()
        self.train()

    def generate_video(self, image_path, prompt, num_frames=81, output_path='output.mp4'):
        try:
            start_image_pil = Image.open(image_path)
            if start_image_pil.mode != 'RGB':
                print(f"Image mode is {start_image_pil.mode}, converting to RGB...")
                start_image_pil = start_image_pil.convert('RGB')
            cprint("start inference...",'green')
            self.inference(
                image=start_image_pil, 
                prompt=prompt,
                negative_prompt="",
                height=self.args.train_resolution[1],
                width=self.args.train_resolution[2],
                num_frames=num_frames,
                output_path=output_path,
                fps=16,
                guidance_scale=5,
            )
        except FileNotFoundError:
            cprint('error','red')

    def decode_latents(self, latents):
        vae = self.components.vae
        if latents.dim() == 4:
            latents = latents.unsqueeze(0)

        latents = latents.to(vae.device, dtype=vae.dtype)
        latents_mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std_inv = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std_inv + latents_mean
        with torch.no_grad():
            video = vae.decode(latents, return_dict=False)[0]
            
        return video

    def inference(
        self,
        prompt,
        image,
        negative_prompt: str = "",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        generator: Optional[torch.Generator] = None,
        output_path: str = "out.mp4",
        fps: int = 8,
    ) -> None:
        """
        Generates a video based on a text prompt and a starting image,
        consistent with Wan 2.2 I2V (`expand_timesteps=False`).
        """
        device = self.accelerator.device
        model_dtype = self.state.weight_dtype

        # components_to_eval = [
        #     "high_noise_model", "low_noise_model", "vae", 
        #     "text_encoder", "tokenizer", "scheduler"
        # ]
        components_to_eval = [ "vae", "text_encoder"]
        for comp_name in components_to_eval:
            comp = getattr(self.components, comp_name)
            if hasattr(comp, "to"):
                comp.to(device)
            if hasattr(comp, "eval"):
                comp.eval()

        self.components.high_noise_model.to("cpu")
        self.components.low_noise_model.to("cpu")
        torch.cuda.empty_cache() 
        
        cprint(f"Running inference on device: {device}", "green")
        
        from torchvision import transforms
        image_transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
        ])
        image_tensor = image_transform(image).unsqueeze(0).to(device, dtype=self.components.vae.dtype)

        prompt_embeds = self.encode_text(prompt).to(device, dtype=model_dtype)
        negative_prompt_embeds = self.encode_text(negative_prompt).to(device, dtype=model_dtype)
        
        vae_scale_factor_spatial = 8
        latent_height = height // vae_scale_factor_spatial
        latent_width = width // vae_scale_factor_spatial
        vae_scale_factor_temporal = 4
        num_latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1

        latents_shape = (1, self.components.vae.config.z_dim, num_latent_frames, latent_height, latent_width)
        latents = torch.randn(latents_shape, generator=generator, device=device, dtype=model_dtype)

        image_for_cond = image_tensor.unsqueeze(2) # [1, C, 1, H, W]
        video_condition = torch.cat(
            [image_for_cond, image_for_cond.new_zeros(1, image_for_cond.shape[1], num_frames - 1, height, width)], dim=2
        )
        video_condition = video_condition.to(device=device, dtype=self.components.vae.dtype)

        with torch.inference_mode():
            latent_condition = self.encode_video(video_condition).to(model_dtype)

        latents_mean = torch.tensor(self.components.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latent_condition)
        latents_std = torch.tensor(self.components.vae.config.latents_std).view(1, -1, 1, 1, 1).to(latent_condition)
        latent_condition = (latent_condition - latents_mean) / latents_std

        mask_lat_size = torch.ones(1, 1, num_frames, latent_height, latent_width, device=device, dtype=model_dtype)
        mask_lat_size[:, :, 1:] = 0
        
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=vae_scale_factor_temporal)
        
        if num_frames > 1:
            mask_lat_size = torch.cat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        else:
            mask_lat_size = first_frame_mask
            
        mask_lat_size = mask_lat_size.view(1, -1, vae_scale_factor_temporal, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2).contiguous()
        
        y_condition = torch.cat([mask_lat_size, latent_condition], dim=1)
        self.components.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.components.scheduler.timesteps

        current_model_on_gpu = None
        for t in tqdm(timesteps, desc="Denoising"):
            model_input = torch.cat([latents, y_condition], dim=1)
            
            if t >= self.state.moe_boundary:
                target_model = self.components.high_noise_model
                other_model = self.components.low_noise_model
            else:
                target_model = self.components.low_noise_model
                other_model = self.components.high_noise_model

            if current_model_on_gpu is not target_model:
                cprint(f"Switching model for timestep {t.item()}", "yellow")
                if current_model_on_gpu is not None:
                    current_model_on_gpu.to("cpu")
                target_model.to(device, dtype=model_dtype)
                current_model_on_gpu = target_model
                torch.cuda.empty_cache()

            t = t.unsqueeze(0) 
            # CFG
            with torch.inference_mode():
                noise_uncond = current_model_on_gpu(model_input, timestep=t, encoder_hidden_states=negative_prompt_embeds)[0]
                noise_cond = current_model_on_gpu(model_input, timestep=t, encoder_hidden_states=prompt_embeds)[0]
                
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            latents = self.components.scheduler.step(noise_pred, t, latents).prev_sample

        if current_model_on_gpu is not None:
            current_model_on_gpu.to("cpu")
        torch.cuda.empty_cache()

        latents_mean = torch.tensor(self.components.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents)
        latents_std = torch.tensor(self.components.vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents)
        latents = latents / (1.0 / latents_std) + latents_mean
        
        with torch.inference_mode():
            video_tensor = self.decode_latents(latents.to(self.components.vae.dtype)).squeeze(0).float()

        
        video_frames_np = (video_tensor / 2 + 0.5).clamp(0, 1)
        video_frames_np = (video_frames_np.permute(1,2,3,0) * 255).cpu().numpy().astype(np.uint8)
        
        from diffusers.utils import export_to_video
        export_to_video(video_frames_np, output_path, fps=fps)
        
        cprint(f"Inference complete. Video saved to {output_path}", "green")


register("wan-i2v", "lora", WanI2VTrainer)