# Modified from: https://github.com/roboterax/video-prediction-policy
# Video Prediction Policy (VPP) architecture adapted from the VPP project;
# extended with RynnWorld4D 3-branch feature extraction (video + depth + flow).

import logging
from typing import Dict, Optional
from torch import nn
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import einops
import torch

from policy_models.edm_diffusion.flow_matching import FlowMatchingPolicy
from policy_models.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler
from policy_models.module.Video_Former import Video_Former_3D
from policy_models.module.wan_feature_extractor import WanFeatureExtractor


logger = logging.getLogger(__name__)


class VPP_Policy(pl.LightningModule):

    def __init__(
            self,
            optimizer: DictConfig,
            lr_scheduler: DictConfig,
            latent_dim: int = 384,
            multistep: int = 10,
            use_lr_scheduler: bool = True,
            act_window_size: int = 10,
            use_text_not_embedding: bool = True,
            seed: int = 42,
            pretrained_model_path: str = '',
            Former_depth: int = 6,
            Former_heads: int = 8,
            Former_dim_head: int = 64,
            Former_num_time_embeds: int = 4,
            num_latents: int = 224,
            use_Former: str = '3d',
            timestep: int = 500,
            max_length: int = 512,
            extract_layer_idx: int = 1,
            use_all_layer: bool = False,
            obs_seq_len: int = 1,
            action_dim: int = 54,
            action_seq_len: int = 10,
            backbone: str = 'rynnworld4d',
            wan_extract_block_idx=15,
            wan_height: int = 224,
            wan_width: int = 224,
            wan_num_frames: int = 16,
            proprio_dim: int = 54,
            num_inference_steps: int = 4,
            goal_seq_len: int = 32,
            text_max_length: int = 77,
            # RynnWorld4D-specific
            rynnworld4d_ckpt: str = "",
            sft_ckpt_path: str = "",
            rynnworld4d_fusion_mode: str = "unidirectional",
            rynnworld4d_share_ffn: bool = False,
            rynnworld4d_zero_fusion: bool = True,
            rynnworld4d_joint_start_layer: int = 0,
            wan_num_inference_layers: int = None,
            da3_quantize: bool = False,
            # Unused (kept for config compat)
            text_encoder_path: str = '',
            use_gripper: bool = False,
            policy_type: str = 'flow_matching',
            use_position_encoding: bool = False,
            sampler_type: str = 'ddim',
            num_sampling_steps: int = 10,
            sigma_data: float = 0.5,
            sigma_min: float = 0.001,
            sigma_max: float = 80,
            noise_scheduler: str = 'exponential',
            sigma_sample_density_type: str = 'loglogistic',
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.act_window_size = act_window_size
        self.action_dim = action_dim
        self.timestep = timestep
        self.extract_layer_idx = extract_layer_idx
        self.use_Former = use_Former
        self.Former_num_time_embeds = Former_num_time_embeds
        self.max_length = max_length
        self.backbone = backbone
        self.proprio_dim = proprio_dim
        self.use_all_layer = use_all_layer
        self.use_text_not_embedding = use_text_not_embedding
        self.goal_seq_len = goal_seq_len
        self.text_max_length = text_max_length

        # --- Condition dim from RynnWorld4D backbone ---
        wan_inner_dim = 3072
        n_layers = len(wan_extract_block_idx) if isinstance(wan_extract_block_idx, (list, tuple)) else 1
        n_branches = 3 if backbone == 'rynnworld4d' else 1
        condition_dim = wan_inner_dim * n_layers * n_branches if use_all_layer else wan_inner_dim * n_branches

        # --- Video_Former (Perceiver resampler) ---
        if use_Former == '3d':
            self.Video_Former = Video_Former_3D(
                dim=latent_dim, depth=Former_depth, dim_head=Former_dim_head,
                heads=Former_heads, num_time_embeds=Former_num_time_embeds,
                num_latents=num_latents, num_frame=Former_num_time_embeds,
                condition_dim=condition_dim, use_temporal=True,
            )
        else:
            self.Video_Former = nn.Linear(condition_dim, latent_dim)

        # --- Language goal encoder: reuse UMT5 from RynnWorld4D pipeline (frozen) ---
        # Text encoding is done via self.TVP_encoder._encode_text()
        # Output: (B, text_max_length, 4096) — we use goal_seq_len tokens for policy head

        # --- RynnWorld4D feature extractor (frozen backbone) ---
        self.TVP_encoder = WanFeatureExtractor(
            wan_pretrained_path=pretrained_model_path,
            extract_block_idx=wan_extract_block_idx,
            use_all_layer=use_all_layer,
            num_frames=wan_num_frames,
            height=wan_height, width=wan_width,
            dtype=torch.bfloat16,
            backbone=backbone,
            rynnworld4d_ckpt=rynnworld4d_ckpt,
            sft_ckpt_path=sft_ckpt_path,
            rynnworld4d_fusion_mode=rynnworld4d_fusion_mode,
            rynnworld4d_share_ffn=rynnworld4d_share_ffn,
            rynnworld4d_zero_fusion=rynnworld4d_zero_fusion,
            rynnworld4d_joint_start_layer=rynnworld4d_joint_start_layer,
            num_inference_layers=wan_num_inference_layers,
            da3_quantize=da3_quantize,
        )
        self.TVP_encoder.pipeline.to(self.device)

        # --- Flow matching policy head ---
        self.model = FlowMatchingPolicy(
            action_dim=action_dim, obs_dim=latent_dim, goal_dim=4096,
            num_tokens=num_latents, goal_window_size=goal_seq_len,
            obs_seq_len=obs_seq_len, act_seq_len=action_seq_len,
            device=self.device, proprio_dim=proprio_dim,
        ).to(self.device)

        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_hyperparameters(ignore=["optimizer", "lr_scheduler"])
        self.num_inference_steps = num_inference_steps
        self.rollout_step_counter = 0
        self.multistep = multistep
        self.latent_goal = None
        self.plan = None
        self.seed = seed
        self.use_lr_scheduler = use_lr_scheduler

        # Freeze unused embeddings
        if proprio_dim == 0:
            for param in self.model.inner_model.proprio_emb.parameters():
                param.requires_grad = False
        self.model.inner_model.pos_emb.requires_grad = False

    def process_device(self):
        self.TVP_encoder.pipeline = self.TVP_encoder.pipeline.to(self.device)

    def configure_optimizers(self):
        optim_groups = [
            {"params": self.model.inner_model.parameters(),
             "weight_decay": self.optimizer_config.transformer_weight_decay},
            {"params": self.Video_Former.parameters(),
             "weight_decay": self.optimizer_config.transformer_weight_decay},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.optimizer_config.learning_rate,
            betas=self.optimizer_config.betas,
        )
        if self.use_lr_scheduler:
            lr_configs = OmegaConf.create(self.lr_scheduler)
            scheduler = TriStageLRScheduler(optimizer, lr_configs)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
        return optimizer

    def on_before_zero_grad(self, optimizer=None):
        total_grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item() ** 2
        self.log("train/grad_norm", total_grad_norm ** 0.5, on_step=True, on_epoch=False, sync_dist=True)

    # ======================== Training / Validation ========================

    def training_step(self, dataset_batch: Dict[str, Dict]) -> torch.Tensor:
        predictive_feature, latent_goal = self.extract_predictive_feature(dataset_batch)
        loss, _ = self.model.loss(predictive_feature, dataset_batch["actions"], latent_goal)
        self.log("train/action_loss", loss, on_step=False, on_epoch=True, sync_dist=True,
                 batch_size=dataset_batch["actions"].shape[0])
        return loss

    @torch.no_grad()
    def validation_step(self, dataset_batch: Dict[str, Dict]) -> Dict[str, torch.Tensor]:
        predictive_feature, latent_goal = self.extract_predictive_feature(dataset_batch)
        B = latent_goal.shape[0]
        action_pred = self.model.sample(
            predictive_feature, latent_goal,
            shape=(B, self.act_window_size, self.action_dim),
            n_steps=self.num_inference_steps,
        )
        pred_loss = torch.nn.functional.mse_loss(
            action_pred, dataset_batch["actions"].to(action_pred.device),
        )
        return {"idx": dataset_batch["idx"], "validation_loss": pred_loss}

    # ======================== Feature Extraction ========================

    def extract_predictive_feature(self, dataset_batch):
        rgb_static = dataset_batch["rgb_obs"]["rgb_static"].to(self.device)
        num_frames = self.Former_num_time_embeds

        depth_cond = dataset_batch.get("depth_static", None)
        if depth_cond is not None:
            depth_cond = depth_cond.to(self.device)

        # Language goal: use pre-computed embedding if available, otherwise encode text
        if "lang_text_embedding" in dataset_batch:
            latent_goal = dataset_batch["lang_text_embedding"].to(self.device)  # (B, seq_len, 4096)
            language = None  # Not needed for feature extraction
        else:
            language = dataset_batch["lang_text"]
            with torch.no_grad():
                latent_goal = self.TVP_encoder._encode_text(language, max_length=self.text_max_length)
        latent_goal = latent_goal[:, :self.goal_seq_len, :].to(rgb_static.dtype)  # (B, goal_seq_len, 4096)

        # Visual features from RynnWorld4D backbone (frozen)
        with torch.no_grad():
            if "lang_text_embedding" in dataset_batch:
                perceptual_features = self.TVP_encoder(
                    rgb_static, None, self.timestep,
                    self.extract_layer_idx, all_layer=self.use_all_layer,
                    step_time=1, max_length=self.max_length,
                    pre_computed_text_emb=latent_goal,
                    depth_cond=depth_cond,
                )
            else:
                perceptual_features = self.TVP_encoder(
                    rgb_static, language, self.timestep,
                    self.extract_layer_idx, all_layer=self.use_all_layer,
                    step_time=1, max_length=self.max_length,
                    depth_cond=depth_cond,
                )

        perceptual_features = einops.rearrange(perceptual_features, 'b f c h w -> b f c (h w)')
        perceptual_features = einops.rearrange(perceptual_features, 'b f c l -> b f l c')
        perceptual_features = perceptual_features[:, :num_frames, :, :]

        # Video_Former compress
        perceptual_features = perceptual_features.to(torch.float32)
        perceptual_features = self.Video_Former(perceptual_features)
        if self.use_Former == 'linear':
            perceptual_features = rearrange(perceptual_features, 'b T q d -> b (T q) d')

        predictive_feature = {'state_images': perceptual_features, 'modality': 'lang'}

        # Proprioception
        if self.proprio_dim > 0 and 'state' in dataset_batch:
            predictive_feature['state_obs'] = dataset_batch['state'].to(self.device).to(torch.float32)

        return predictive_feature, latent_goal

    # ======================== Inference ========================

    def eval_forward(self, obs, goal):
        rgb_static = obs["rgb_obs"]["rgb_static"].to(self.device)
        num_frames = self.Former_num_time_embeds

        depth_cond = obs.get("depth_static", None)
        if depth_cond is not None:
            depth_cond = depth_cond.to(self.device)

        # Support both raw text and pre-computed embeddings
        pre_computed_emb = None
        if "lang_text_embedding" in goal:
            # Pre-computed UMT5 embedding (seq_len, 4096) - batch added by collate
            pre_computed_emb = goal["lang_text_embedding"].to(self.device).to(self.dtype)
            if pre_computed_emb.dim() == 2:
                pre_computed_emb = pre_computed_emb.unsqueeze(0)
            latent_goal = pre_computed_emb[:, :self.goal_seq_len, :].to(torch.float32)
        else:
            # Raw text -> encode via UMT5
            language = goal["lang_text"]
            with torch.no_grad():
                latent_goal = self.TVP_encoder._encode_text(
                    language if isinstance(language, list) else [language],
                    max_length=self.text_max_length,
                )
            latent_goal = latent_goal[:, :self.goal_seq_len, :].to(torch.float32)

        with torch.no_grad():
            perceptual_features = self.TVP_encoder(
                rgb_static, ["goal"],  # placeholder, ignored if pre_computed_emb provided
                self.timestep, self.extract_layer_idx,
                all_layer=self.use_all_layer, step_time=1, max_length=self.max_length,
                pre_computed_text_emb=pre_computed_emb,
                depth_cond=depth_cond,
            )

        perceptual_features = einops.rearrange(perceptual_features, 'b f c h w -> b f c (h w)')
        perceptual_features = einops.rearrange(perceptual_features, 'b f c l -> b f l c')
        perceptual_features = perceptual_features[:, :num_frames, :, :]
        perceptual_features = perceptual_features.to(torch.float32)
        perceptual_features = self.Video_Former(perceptual_features)

        perceptual_emb = {'state_images': perceptual_features, 'modality': 'lang'}
        if 'state' in obs:
            perceptual_emb['state_obs'] = obs['state'].to(self.device).to(torch.float32)

        B = latent_goal.shape[0]
        return self.model.sample(
            perceptual_emb, latent_goal,
            shape=(B, self.act_window_size, self.action_dim),
            n_steps=self.num_inference_steps,
        )

    def step(self, obs, goal):
        if self.rollout_step_counter % self.multistep == 0:
            self.pred_action_seq = self.eval_forward(obs, goal)
        current_action = self.pred_action_seq[0, self.rollout_step_counter]
        if len(current_action.shape) == 2:
            current_action = einops.rearrange(current_action, 'b d -> b 1 d')
        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.multistep:
            self.rollout_step_counter = 0
        return current_action

    def reset(self):
        self.plan = None
        self.latent_goal = None
        self.rollout_step_counter = 0

    def forward(self, batch):
        return self.training_step(batch)

    def on_train_start(self):
        self.model.to(dtype=self.dtype)
        self.Video_Former.to(dtype=self.dtype)

    @rank_zero_only
    def on_train_epoch_start(self):
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused=None):
        logger.info(f"Finished training epoch {self.current_epoch}")

    def on_validation_epoch_start(self):
        logger.info(f"Start validation epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_end(self):
        logger.info(f"Finished validation epoch {self.current_epoch}")
