import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from safetensors.torch import load_file
from termcolor import cprint
from torchvision import transforms

from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanImageToVideoPipeline,
    WanTransformer3DModel,
)
from diffusers.utils import BaseOutput, export_to_video
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from core.finetune.models.wan_i2v.module import (
    RynnWorld4DTransformer3DModel,
    patched_wan_time_text_image_embedding_forward,
)

logging.basicConfig(level=logging.INFO)

try:
    from diffusers.models.transformers.transformer_wan import WanTimeTextImageEmbedding
except ImportError as e:
    cprint("Could not import WanTimeTextImageEmbedding for monkey-patching.", "red")
    raise e

WanTimeTextImageEmbedding.forward = patched_wan_time_text_image_embedding_forward


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List[torch.Generator], torch.Generator]] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
):
    if isinstance(device, str):
        device = torch.device(device)
    rand_device = device
    batch_size = shape[0]
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"

    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)

    return latents


class RynnWorld4DPipeline(WanImageToVideoPipeline):
    """RynnWorld4D inference pipeline: simultaneously generates RGB, depth, and optical flow videos."""

    @torch.no_grad()
    def encode_first_frame(self, image: Image.Image, height: int, width: int) -> torch.Tensor:
        """Encode a PIL image into a single-frame VAE latent [1, C, 1, H_l, W_l]."""
        image = image.resize((width, height))
        image_tensor = transforms.ToTensor()(image)  # [C, H, W] in [0, 1]
        image_tensor = image_tensor * 2.0 - 1.0  # normalize to [-1, 1]
        # [1, C, 1, H, W]
        video_input = image_tensor.unsqueeze(0).unsqueeze(2)
        video_input = video_input.to(device=self.vae.device, dtype=self.vae.dtype)

        latent = self.vae.encode(video_input).latent_dist.mode()

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latent.device, latent.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latent.device, latent.dtype
        )
        latent = (latent - latents_mean) * latents_std
        return latent  # [1, C_z, 1, H_l, W_l]

    @torch.no_grad()
    def make_zero_flow_latent(self, height: int, width: int) -> torch.Tensor:
        """Encode a white image (zero optical flow in Middlebury) into a single-frame VAE latent."""
        white_image = Image.new("RGB", (width, height), (255, 255, 255))
        return self.encode_first_frame(white_image, height, width)

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image,
        depth_image: Image.Image,
        prompt: str = "",
        negative_prompt: str = "",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        max_sequence_length: int = 512,
        output_type: str = "np",
        return_dict: bool = True,
    ):
        device = self._execution_device
        transformer_dtype = self.transformer.dtype

        # 1. Encode text
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 2. Encode first-frame conditions
        self.vae.to(device)
        img_latent = self.encode_first_frame(image, height, width)           # [1, C, 1, H_l, W_l]
        depth_latent = self.encode_first_frame(depth_image, height, width)   # [1, C, 1, H_l, W_l]
        flow_latent = self.make_zero_flow_latent(height, width)              # [1, C, 1, H_l, W_l]

        img_latent = img_latent.to(device=device, dtype=transformer_dtype)
        depth_latent = depth_latent.to(device=device, dtype=transformer_dtype)
        flow_latent = flow_latent.to(device=device, dtype=transformer_dtype)

        # 3. Prepare noise latents
        batch_size = 1
        num_channels_latents = self.vae.config.z_dim
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        latents_video = randn_tensor(shape, generator=generator, device=device, dtype=transformer_dtype)
        latents_depth = randn_tensor(shape, generator=generator, device=device, dtype=transformer_dtype)
        latents_flow = randn_tensor(shape, generator=generator, device=device, dtype=transformer_dtype)

        # 4. First-frame mask (0 at frame 0, 1 elsewhere) — matches training
        first_frame_mask = torch.ones(
            1, 1, num_latent_frames, latent_height, latent_width,
            device=device, dtype=transformer_dtype,
        )
        first_frame_mask[:, :, 0] = 0

        # 5. Scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                # Set first-frame condition (clean latent, no noise)
                latents_video[:, :, 0:1, :, :] = img_latent
                latents_depth[:, :, 0:1, :, :] = depth_latent
                latents_flow[:, :, 0:1, :, :] = flow_latent

                input_video = latents_video.to(transformer_dtype)
                input_depth = latents_depth.to(transformer_dtype)
                input_flow = latents_flow.to(transformer_dtype)

                # Per-token timestep: 0 for first frame, t for the rest (matches training)
                temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
                timestep_input = temp_ts.unsqueeze(0).expand(batch_size, -1).to(transformer_dtype)

                # Conditional forward
                video_pred, depth_pred, flow_pred = self.transformer(
                    hidden_states=input_video,
                    hidden_states_depth=input_depth,
                    hidden_states_flow=input_flow,
                    timestep=timestep_input,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=None,
                    attention_kwargs=None,
                    return_dict=False,
                )

                # CFG: unconditional forward
                if self.do_classifier_free_guidance:
                    video_uncond, depth_uncond, flow_uncond = self.transformer(
                        hidden_states=input_video,
                        hidden_states_depth=input_depth,
                        hidden_states_flow=input_flow,
                        timestep=timestep_input,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_image=None,
                        attention_kwargs=None,
                        return_dict=False,
                    )
                    video_pred = video_uncond + guidance_scale * (video_pred - video_uncond)
                    depth_pred = depth_uncond + guidance_scale * (depth_pred - depth_uncond)
                    flow_pred = flow_uncond + guidance_scale * (flow_pred - flow_uncond)

                # Scheduler step
                latents_video = self.scheduler.step(video_pred, t, latents_video, return_dict=False)[0]
                latents_depth = self.scheduler.step(depth_pred, t, latents_depth, return_dict=False)[0]
                latents_flow = self.scheduler.step(flow_pred, t, latents_flow, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None

        # 7. Final first-frame replacement
        latents_video[:, :, 0:1, :, :] = img_latent
        latents_depth[:, :, 0:1, :, :] = depth_latent
        latents_flow[:, :, 0:1, :, :] = flow_latent

        # 8. VAE decode
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(device=device, dtype=self.vae.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            device=device, dtype=self.vae.dtype
        )

        def decode_latent(lat):
            lat = lat.to(self.vae.dtype)
            lat = lat / latents_std + latents_mean
            video = self.vae.decode(lat, return_dict=False)[0]
            return self.video_processor.postprocess_video(video, output_type=output_type)

        video_rgb = decode_latent(latents_video)
        video_depth = decode_latent(latents_depth)
        video_flow = decode_latent(latents_flow)

        self.maybe_free_model_hooks()

        if not return_dict:
            return video_rgb, video_depth, video_flow

        return (
            BaseOutput(frames=video_rgb),
            BaseOutput(frames=video_depth),
            BaseOutput(frames=video_flow),
        )


def generate_rynnworld4d_video(
    model_path: str,
    lora_path: str,
    image_path: str,
    depth_image_path: str,
    prompt: str = "",
    negative_prompt: str = "",
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    seed: int = 42,
    output_dir: str = "results",
    output_prefix: str = "rynnworld4d",
    dtype: torch.dtype = torch.bfloat16,
):
    device = torch.device("cuda")

    # 1. Load pipeline with RynnWorld4DTransformer3DModel
    cprint("Loading RynnWorld4D pipeline...", "cyan")
    transformer = RynnWorld4DTransformer3DModel.from_pretrained(
        model_path, subfolder="transformer", eps=1e-5
    )

    pipe = RynnWorld4DPipeline.from_pretrained(
        model_path,
        transformer=transformer,
        torch_dtype=dtype,
    )

    # 2. Load and fuse LoRA weights, then remove PEFT wrappers
    high_noise_lora_dir = os.path.join(lora_path, "high_noise_lora")
    if os.path.exists(high_noise_lora_dir):
        pipe.load_lora_weights(
            high_noise_lora_dir,
            weight_name="pytorch_lora_weights.safetensors",
        )
        pipe.fuse_lora(components=["transformer"], lora_scale=1)
        pipe.unload_lora_weights()
        cprint("Loaded and fused LoRA weights.", "green")

    # 3. Load RynnWorld4D branch weights (depth/flow specific layers)
    # Must be done AFTER LoRA fuse+unload so keys are clean and depth/flow
    # branch parameters override correctly.
    rynnworld4d_layers_path = os.path.join(lora_path, "rynnworld4d_layers.bin")
    if os.path.exists(rynnworld4d_layers_path):
        saved_weights = torch.load(rynnworld4d_layers_path, map_location="cpu")
        cleaned_weights = {}
        for k, v in saved_weights.items():
            new_k = k.replace(".base_layer.", ".")
            cleaned_weights[new_k] = v
        missing, unexpected = pipe.transformer.load_state_dict(cleaned_weights, strict=False)
        important_missing = [k for k in missing if ("depth" in k or "flow" in k)]
        if important_missing:
            cprint(f"WARNING: Missing depth/flow keys: {important_missing[:10]}", "red")
        cprint(f"Loaded RynnWorld4D branch weights ({len(cleaned_weights)} params).", "green")

    pipe.enable_model_cpu_offload()

    # 4. Load input images
    image = Image.open(image_path).convert("RGB")
    depth_image = Image.open(depth_image_path).convert("RGB")

    generator = torch.Generator(device="cpu").manual_seed(seed)

    # 5. Run inference
    cprint("Running RynnWorld4D inference...", "cyan")
    output_rgb, output_depth, output_flow = pipe(
        image=image,
        depth_image=depth_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    # 6. Save outputs
    os.makedirs(output_dir, exist_ok=True)

    rgb_path = os.path.join(output_dir, f"{output_prefix}_rgb.mp4")
    depth_path = os.path.join(output_dir, f"{output_prefix}_depth.mp4")
    flow_path = os.path.join(output_dir, f"{output_prefix}_flow.mp4")

    export_to_video(output_rgb.frames[0], rgb_path, fps=16)
    export_to_video(output_depth.frames[0], depth_path, fps=16)
    export_to_video(output_flow.frames[0], flow_path, fps=16)

    cprint(f"Saved RGB video:   {rgb_path}", "green")
    cprint(f"Saved depth video: {depth_path}", "green")
    cprint(f"Saved flow video:  {flow_path}", "green")

    return rgb_path, depth_path, flow_path
