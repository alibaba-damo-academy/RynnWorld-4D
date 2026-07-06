"""
Wan2.2-TI2V-5B feature extractor for VPP.

Modified from: https://github.com/roboterax/video-prediction-policy
Feature extraction interface adapted from VPP; extended with RynnWorld4D
3-branch transformer and Depth-Anything-3 depth estimation.

Supports two backends:
  * backbone="wan"       — standard WanPipeline (single branch)
  * backbone="rynnworld4d" — RynnWorld4D 3-branch model (video + depth + flow)

Both expose the same calling convention so VPP_policy can use them
interchangeably:

    feats = extractor(pixel_values, texts, timestep, ...)
    # feats shape: (B, F_lat, condition_dim, H_lat, W_lat)
"""
import gc
import os
import sys
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
from einops import rearrange


class _StopForward(Exception):
    """Sentinel raised from a forward hook to abort the transformer forward
    pass early once all required block outputs have been captured."""
    pass


class WanFeatureExtractor(nn.Module):
    """Single-step feature extractor wrapping Wan2.2 transformer.

    When backbone="rynnworld4d", loads RynnWorld4DTransformer3DModel and
    extracts features from all three branches (video+depth+flow),
    concatenated along the spatial-token dimension.  Video_Former
    (Perceiver resampler) handles the 3× token increase natively.

    Args:
        wan_pretrained_path: path to Wan2.2-TI2V-5B-Diffusers folder.
        extract_block_idx: block index(es) whose output is extracted.
        use_all_layer: if True and multiple indices, channel-cat them.
        num_frames: pixel frames fed to VAE (typically 16).
        height/width: pixel input resolution.
        dtype: compute dtype for transformer forward.
        backbone: "wan" (standard) or "rynnworld4d" (3-branch).
        rynnworld4d_ckpt: path to RynnWorld4D SFT checkpoint directory.
        rynnworld4d_fusion_mode: "none" / "unidirectional" / "bidirectional".
        rynnworld4d_share_ffn: whether depth/flow share FFN with video.
        rynnworld4d_zero_fusion: zero out fusion layers (stage-1 ckpts).
    """

    SVD_NUM_FRAMES_DEFAULT = 16

    def __init__(
        self,
        wan_pretrained_path: str,
        extract_block_idx: Union[int, Sequence[int]] = 15,
        use_all_layer: bool = False,
        num_frames: int = 16,
        height: int = 256,
        width: int = 256,
        dtype: torch.dtype = torch.bfloat16,
        # RynnWorld4D-specific
        backbone: str = "wan",
        rynnworld4d_ckpt: str = "",
        sft_ckpt_path: str = "",
        rynnworld4d_fusion_mode: str = "unidirectional",
        rynnworld4d_share_ffn: bool = False,
        rynnworld4d_zero_fusion: bool = True,
        rynnworld4d_joint_start_layer: int = 0,
        rynnworld4d_joint_end_layer: int = -1,
        rynnworld4d_joint_every_n_layers: int = 1,
        rynnworld4d_joint_frame_wise: bool = False,
        rynnworld4d_joint_use_rope: bool = False,
        rynnworld4d_joint_unidirectional: bool = False,
        depth_model_path: str = "",  # path to Depth-Anything-3 weights (e.g. ./pretrained/da3)
        num_inference_layers: Optional[int] = None,
        da3_quantize: bool = False,
    ):
        super().__init__()

        self.dtype = dtype
        self.da3_quantize = da3_quantize
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.backbone_type = backbone  # "wan" or "rynnworld4d"
        self.depth_model_path = depth_model_path
        self._da3_model = None
        self._da3_quantized = False

        if backbone == "rynnworld4d":
            self._init_rynnworld4d(
                wan_pretrained_path, rynnworld4d_ckpt, sft_ckpt_path,
                rynnworld4d_fusion_mode, rynnworld4d_share_ffn, rynnworld4d_zero_fusion,
                rynnworld4d_joint_start_layer,
                dtype,
                joint_end_layer=rynnworld4d_joint_end_layer,
                joint_every_n_layers=rynnworld4d_joint_every_n_layers,
                joint_frame_wise=rynnworld4d_joint_frame_wise,
                joint_use_rope=rynnworld4d_joint_use_rope,
                joint_unidirectional=rynnworld4d_joint_unidirectional,
            )
        else:
            self._init_wan(wan_pretrained_path, dtype)

        # Common config from transformer
        cfg = self.transformer.config
        self.inner_dim = cfg.num_attention_heads * cfg.attention_head_dim  # 3072
        self.num_blocks = cfg.num_layers  # 30 or 40
        self.patch_size = tuple(cfg.patch_size)  # (1, 2, 2)

        # VAE latent normalization
        vae_cfg = self.vae.config
        latents_mean = torch.tensor(vae_cfg.latents_mean).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(vae_cfg.latents_std).view(1, -1, 1, 1, 1)
        self.register_buffer("latents_mean", latents_mean)
        self.register_buffer("latents_std", latents_std)

        # Extract block indices
        if isinstance(extract_block_idx, int):
            self.extract_block_idx = [extract_block_idx]
        else:
            self.extract_block_idx = list(extract_block_idx)
        for idx in self.extract_block_idx:
            assert 0 <= idx < self.num_blocks, f"extract_block_idx={idx} out of range"
        self.use_all_layer = use_all_layer

        # Early-exit: only run the first `num_inference_layers` transformer
        # blocks. Layers beyond the deepest extracted block are pure waste, so
        # truncating the forward saves compute/latency without affecting the
        # extracted features (as long as the cutoff covers max(extract_block_idx)).
        min_required = max(self.extract_block_idx) + 1
        if num_inference_layers is None:
            self.num_inference_layers = self.num_blocks
        else:
            if num_inference_layers < min_required:
                print(
                    f"[WanFeatureExtractor] num_inference_layers={num_inference_layers} "
                    f"< required {min_required} (max extract_block_idx + 1); "
                    f"clamping to {min_required}"
                )
                num_inference_layers = min_required
            self.num_inference_layers = min(num_inference_layers, self.num_blocks)
        # Index of the deepest block we actually need to execute (0-based).
        self.last_needed_block = self.num_inference_layers - 1
        if self.num_inference_layers < self.num_blocks:
            print(
                f"[WanFeatureExtractor] Early-exit enabled: running "
                f"{self.num_inference_layers}/{self.num_blocks} transformer blocks"
            )
            # Physically drop the unused trailing blocks. The transformer
            # forward iterates `for block in self.blocks`, so removing them
            # skips their compute AND prevents them from ever being moved to
            # GPU (also frees their CPU memory).
            if hasattr(self.transformer, "blocks"):
                kept = nn.ModuleList(
                    list(self.transformer.blocks)[: self.num_inference_layers]
                )
                n_dropped = len(self.transformer.blocks) - len(kept)
                self.transformer.blocks = kept
                if hasattr(self.transformer, "config"):
                    try:
                        self.transformer.config.num_layers = self.num_inference_layers
                    except Exception:
                        pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(
                    f"[WanFeatureExtractor] Dropped {n_dropped} trailing transformer "
                    f"blocks (not loaded to GPU)"
                )

        # condition_dim exposed to downstream Video_Former
        n_branches = 3 if backbone == "rynnworld4d" else 1
        if self.use_all_layer:
            self.condition_dim = self.inner_dim * len(self.extract_block_idx) * n_branches
        else:
            self.condition_dim = self.inner_dim * n_branches

        # Freeze everything
        for p in self.parameters():
            p.requires_grad = False

    # ====================== init helpers ======================

    def _init_wan(self, pretrained_path, dtype):
        """Standard single-branch Wan pipeline."""
        from diffusers import WanPipeline

        pipeline = WanPipeline.from_pretrained(pretrained_path, torch_dtype=dtype)
        self.pipeline = pipeline
        self.vae = pipeline.vae
        self.transformer = pipeline.transformer
        self.tokenizer = pipeline.tokenizer
        self.text_encoder = pipeline.text_encoder
        self.scheduler = pipeline.scheduler

    def _init_rynnworld4d(self, pretrained_path, ckpt_path, sft_ckpt_path, fusion_mode, share_ffn, zero_fusion, joint_start_layer, dtype,
                          joint_end_layer=-1, joint_every_n_layers=1, joint_frame_wise=False, joint_use_rope=False, joint_unidirectional=False):
        """RynnWorld4D 3-branch transformer + standard Wan VAE/tokenizer."""
        # Locate the RynnWorld4D project root (parent of this rynnworld4d_policy package).
        _here = os.path.abspath(__file__)
        rynnworld4d_root = os.path.abspath(os.path.join(_here, "..", "..", "..", ".."))
        if not os.path.isdir(os.path.join(rynnworld4d_root, "core")):
            raise RuntimeError(
                f"Could not locate RynnWorld4D project root from {_here}; "
                f"expected `core/` at {rynnworld4d_root}. Run this from a checkout of the repo."
            )
        if rynnworld4d_root not in sys.path:
            sys.path.insert(0, rynnworld4d_root)

        from diffusers import WanPipeline
        from diffusers.models.transformers.transformer_wan import WanTimeTextImageEmbedding
        from core.finetune.models.wan_i2v.module import (
            RynnWorld4DTransformer3DModel,
            patched_wan_time_text_image_embedding_forward,
        )

        # Apply monkey-patch for float32 stability
        WanTimeTextImageEmbedding.forward = patched_wan_time_text_image_embedding_forward

        # Load RynnWorld4D transformer from base Wan weights
        if fusion_mode == "joint":
            from core.finetune.models.wan_i2v.module_joint import JointRynnWorld4DTransformer3DModel
            transformer = JointRynnWorld4DTransformer3DModel.from_pretrained(
                pretrained_path, subfolder="transformer", torch_dtype=dtype, eps=1e-5,
                share_ffn=share_ffn,
                joint_start_layer=joint_start_layer,
                joint_end_layer=joint_end_layer,
                joint_every_n_layers=joint_every_n_layers,
                joint_frame_wise=joint_frame_wise,
                joint_use_rope=joint_use_rope,
                joint_unidirectional=joint_unidirectional,
            )
        else:
            transformer = RynnWorld4DTransformer3DModel.from_pretrained(
                pretrained_path, subfolder="transformer", torch_dtype=dtype, eps=1e-5,
                share_ffn=share_ffn, fusion_mode=fusion_mode,
            )

        # Load SFT checkpoint BEFORE moving to GPU / creating pipeline
        # This keeps peak CPU memory lower since the transformer isn't duplicated on GPU yet
        sft_path = sft_ckpt_path if sft_ckpt_path else ckpt_path
        if sft_path and os.path.exists(sft_path):
            self._load_sft_checkpoint(sft_path, transformer)

        # Now move to target dtype (also moves to GPU later via pipeline)
        transformer.to(dtype)

        # Zero fusion layers if needed (stage-1 checkpoints)
        if zero_fusion:
            self._zero_fusion_layers(transformer)

        # Load rest of pipeline (VAE, tokenizer, scheduler). Skip the ~11GB UMT5
        # text_encoder: deployment uses pre-computed text embeddings, and
        # _encode_text is only invoked when no embedding is supplied.
        pipeline = WanPipeline.from_pretrained(
            pretrained_path, transformer=transformer, text_encoder=None, torch_dtype=dtype
        )
        self.pipeline = pipeline
        self.vae = pipeline.vae
        self.transformer = transformer
        self.tokenizer = pipeline.tokenizer
        self.text_encoder = pipeline.text_encoder  # None when skipped
        self.scheduler = pipeline.scheduler

    @staticmethod
    def _load_sft_checkpoint(ckpt_path, transformer):
        """Load RynnWorld4D SFT checkpoint from sharded safetensors (memory-efficient mmap)."""
        from safetensors import safe_open

        index_path = os.path.join(ckpt_path, "pytorch_model", "sft_model.safetensors.index.json")
        if not os.path.exists(index_path):
            # Fall back to original DeepSpeed checkpoint
            model_states_path = os.path.join(ckpt_path, "pytorch_model", "mp_rank_00_model_states.pt")
            if not os.path.exists(model_states_path):
                print(f"[WanFeatureExtractor] SFT checkpoint not found, skipping")
                return
            print(f"[WanFeatureExtractor] No safetensors index, falling back to DeepSpeed loading")
            device = next(transformer.parameters()).device
            checkpoint = torch.load(model_states_path, map_location=device, weights_only=False)
            state_dict = checkpoint.pop("module")
            del checkpoint
            gc.collect()
            torch.cuda.empty_cache()
            keys_to_rename = []
            for k in list(state_dict.keys()):
                new_k = k.replace("module.", "").replace(".base_layer.", ".")
                if new_k != k:
                    keys_to_rename.append((k, new_k))
            for old_k, new_k in keys_to_rename:
                state_dict[new_k] = state_dict.pop(old_k)
            missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
            print(f"  Loaded: {len(state_dict)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
            del state_dict
            gc.collect()
            torch.cuda.empty_cache()
            return

        # Load from safetensors shards using mmap
        print(f"[WanFeatureExtractor] Loading SFT checkpoint from safetensors shards")

        import json
        with open(index_path) as f:
            index = json.load(f)

        weight_map = index["weight_map"]
        shard_files = set(weight_map.values())
        shard_dir = os.path.join(ckpt_path, "pytorch_model")

        # Verify all shard files exist before loading
        missing_shards = [s for s in shard_files if not os.path.exists(os.path.join(shard_dir, s))]
        if missing_shards:
            print(f"[WanFeatureExtractor] WARNING: {len(missing_shards)} safetensors shards missing, falling back to DeepSpeed")
            print(f"  Missing: {missing_shards[:3]}...")
            # Fall through to DeepSpeed loading below
            model_states_path = os.path.join(ckpt_path, "pytorch_model", "mp_rank_00_model_states.pt")
            if not os.path.exists(model_states_path):
                print(f"[WanFeatureExtractor] SFT checkpoint not found, skipping")
                return
            print(f"[WanFeatureExtractor] Falling back to DeepSpeed loading")
            device = next(transformer.parameters()).device
            checkpoint = torch.load(model_states_path, map_location=device, weights_only=False)
            state_dict = checkpoint.pop("module")
            del checkpoint
            gc.collect()
            torch.cuda.empty_cache()
            keys_to_rename = []
            for k in list(state_dict.keys()):
                new_k = k.replace("module.", "").replace(".base_layer.", ".")
                if new_k != k:
                    keys_to_rename.append((k, new_k))
            for old_k, new_k in keys_to_rename:
                state_dict[new_k] = state_dict.pop(old_k)
            missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
            print(f"  Loaded: {len(state_dict)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
            del state_dict
            gc.collect()
            torch.cuda.empty_cache()
            return

        print(f"  Found {len(shard_files)} safetensors shards")

        # Cache state_dict to avoid creating new OrderedDict on every iteration
        model_state = transformer.state_dict()

        count = 0
        for shard_name in shard_files:
            shard_path = os.path.join(shard_dir, shard_name)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    target_key = key.replace("module.", "").replace(".base_layer.", ".")
                    if target_key in model_state:
                        model_state[target_key].copy_(tensor)
                        count += 1

        print(f"  Loaded: {count} keys from {len(shard_files)} safetensors shards")
        gc.collect()

    @staticmethod
    def _zero_fusion_layers(transformer):
        """Zero out fusion layers (for stage-1 SFT where fusion is not trained)."""
        count = 0
        for block in transformer.blocks:
            for name in ["video_to_depth_zero", "video_to_flow_zero",
                         "depth_to_video_zero", "flow_to_video_zero"]:
                layer = getattr(block, name, None)
                if layer is not None:
                    layer.weight.data.zero_()
                    layer.bias.data.zero_()
                    count += 1
        print(f"[WanFeatureExtractor] Zeroed {count} fusion layers")

    # ====================== helpers ======================

    def _encode_text(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """Tokenize + UMT5-encode. Returns (B, max_length, 4096)."""
        if self.text_encoder is None:
            raise RuntimeError(
                "text_encoder was not loaded (skipped to save memory). "
                "Provide a pre-computed 'lang_text_embedding' in the goal/batch "
                "instead of raw 'lang_text'."
            )
        device = self.transformer.device
        inputs = self.tokenizer(
            texts, padding="max_length", max_length=max_length,
            truncation=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = self.text_encoder(**inputs)
        return out.last_hidden_state.to(self.dtype)

    @torch.no_grad()
    def _vae_encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode video pixels to Wan latent. (B, F, C, H, W) -> (B, C_z, F_lat, H_lat, W_lat)"""
        x = pixel_values.permute(0, 2, 1, 3, 4).to(self.vae.dtype)
        posterior = self.vae.encode(x).latent_dist
        return posterior.sample()

    @torch.no_grad()
    def _vae_encode_single_frame(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a single RGB frame per RynnWorld4D convention.
        Args:
            image: (B, C, H, W) in [-1, 1]
        Returns:
            latent: (B, C_z, 1, H_l, W_l) normalized
        """
        x = image.unsqueeze(2).to(self.vae.dtype)  # (B, C, 1, H, W)
        latent = self.vae.encode(x).latent_dist.mode()
        latent = (latent.float() - self.latents_mean.to(latent.device)) / self.latents_std.to(latent.device)
        return latent

    # --------- transformer step (standard Wan) ---------

    @torch.no_grad()
    def _transformer_step(self, latent, text_emb, timestep):
        """Single forward through standard Wan transformer, capture block outputs."""
        device = self.transformer.device
        captured = {idx: None for idx in self.extract_block_idx}

        def make_hook(idx):
            def hook(module, inp, out):
                # Standard Wan block returns single tensor
                captured[idx] = out[0] if isinstance(out, tuple) else out
            return hook

        handles = [
            self.transformer.blocks[idx].register_forward_hook(make_hook(idx))
            for idx in self.extract_block_idx
        ]

        # Early-exit hook: abort once the deepest needed block has produced its
        # output. Registered last so capture hooks on the same block run first.
        def stop_hook(module, inp, out):
            raise _StopForward
        handles.append(
            self.transformer.blocks[self.last_needed_block].register_forward_hook(stop_hook)
        )

        ts = torch.tensor([int(timestep)], device=device, dtype=torch.long)
        try:
            self.transformer(
                hidden_states=latent.to(self.dtype),
                timestep=ts,
                encoder_hidden_states=text_emb,
                return_dict=False,
            )
        except _StopForward:
            pass
        finally:
            for h in handles:
                h.remove()

        return [captured[idx] for idx in self.extract_block_idx]

    # --------- transformer step (RynnWorld4D 3-branch) ---------

    @torch.no_grad()
    def _transformer_step_rynnworld4d(self, latent_video, latent_depth, latent_flow, text_emb, timestep):
        """Single forward through RynnWorld4D transformer, capture 3-branch block outputs.

        Returns list of tensors, each (B, 3*N, D) — video+depth+flow tokens concatenated.
        """
        device = self.transformer.device
        captured = {idx: None for idx in self.extract_block_idx}

        def make_hook(idx):
            def hook(module, inp, out):
                video_out, depth_out, flow_out = out
                captured[idx] = torch.cat([video_out, depth_out, flow_out], dim=1)
            return hook

        handles = [
            self.transformer.blocks[idx].register_forward_hook(make_hook(idx))
            for idx in self.extract_block_idx
        ]

        # Early-exit hook: abort once the deepest needed block has produced its
        # output. Registered last so capture hooks on the same block run first.
        def stop_hook(module, inp, out):
            raise _StopForward
        handles.append(
            self.transformer.blocks[self.last_needed_block].register_forward_hook(stop_hook)
        )

        B = latent_video.shape[0]
        F_lat, H_lat, W_lat = latent_video.shape[2], latent_video.shape[3], latent_video.shape[4]
        p_t, p_h, p_w = self.patch_size

        first_frame_mask = torch.ones(1, 1, F_lat, H_lat, W_lat, device=device, dtype=self.dtype)
        first_frame_mask[:, :, 0] = 0
        temp_ts = (first_frame_mask[0][0][:, ::p_h, ::p_w] * int(timestep)).flatten()
        ts = temp_ts.unsqueeze(0).expand(B, -1)

        try:
            self.transformer(
                hidden_states=latent_video.to(self.dtype),
                hidden_states_depth=latent_depth.to(self.dtype),
                hidden_states_flow=latent_flow.to(self.dtype),
                timestep=ts,
                encoder_hidden_states=text_emb,
                encoder_hidden_states_image=None,
                return_dict=False,
            )
        except _StopForward:
            pass
        finally:
            for h in handles:
                h.remove()

        return [captured[idx] for idx in self.extract_block_idx]

    # ====================== main API ======================

    @torch.no_grad()
    def forward(
        self,
        pixel_values: torch.Tensor,
        texts: List[str],
        timestep: Union[int, torch.Tensor] = 500,
        extract_layer_idx: Optional[int] = None,
        use_latent: bool = False,
        all_layer: Optional[bool] = None,
        step_time: int = 1,
        max_length: int = 512,
        pre_computed_text_emb: Optional[torch.Tensor] = None,
        depth_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run extractor.

        For RynnWorld4D backbone:
            pixel_values: (B, F, C, H, W) — only frame 0 is used as condition.
            Internally constructs 21 latent frames: frame 0 = VAE(condition),
            frames 1-20 = random noise. Matches RynnWorld4D training/inference.
            depth_cond: optional precomputed frame-0 depth image in [-1, 1],
            shape (B, 1, 3, H, W) or (B, 3, H, W). If None, depth is estimated
            online with DA3.

        Returns:
            features of shape (B, F_tok, condition_dim, H_tok, W_tok)
        """
        device = self.transformer.device
        if isinstance(timestep, torch.Tensor):
            timestep = int(timestep.item())

        # Step 1: encode text (or use pre-computed)
        if pre_computed_text_emb is not None:
            text_emb = pre_computed_text_emb.to(device).to(self.dtype)
        else:
            text_emb = self._encode_text(texts, max_length=max_length)

        if self.backbone_type == "rynnworld4d":
            latent_video, latent_depth, latent_flow = self._build_rynnworld4d_latents(
                pixel_values, depth_cond=depth_cond
            )
            block_outs = self._transformer_step_rynnworld4d(latent_video, latent_depth, latent_flow, text_emb, timestep)
            z = latent_video  # for shape computation below
        else:
            # Standard Wan: pad/repeat frames then VAE encode all
            if pixel_values.shape[1] < self.num_frames:
                n_repeat = self.num_frames // pixel_values.shape[1]
                pixel_values = pixel_values.repeat_interleave(n_repeat, dim=1)
                if pixel_values.shape[1] < self.num_frames:
                    pad = self.num_frames - pixel_values.shape[1]
                    pixel_values = torch.cat(
                        [pixel_values, pixel_values[:, -1:].repeat(1, pad, 1, 1, 1)], dim=1,
                    )
            z = self._vae_encode(pixel_values)
            z = (z.float() - self.latents_mean.to(z.device)) / self.latents_std.to(z.device)
            z = z.to(self.dtype)
            block_outs = self._transformer_step(z, text_emb, timestep)

        # Reshape to (B, F_tok, C, H_tok, W_tok)
        B = z.shape[0]
        F_lat, H_lat, W_lat = z.shape[2], z.shape[3], z.shape[4]
        p_t, p_h, p_w = self.patch_size
        F_tok, H_tok, W_tok = F_lat // p_t, H_lat // p_h, W_lat // p_w

        feats = []
        for out in block_outs:
            if self.backbone_type == "rynnworld4d":
                n_tokens_per_branch = F_tok * H_tok * W_tok
                branches = out.split(n_tokens_per_branch, dim=1)
                branch_feats = []
                for br in branches:
                    f = rearrange(br, "b (f h w) c -> b f c h w", f=F_tok, h=H_tok, w=W_tok)
                    branch_feats.append(f)
                f = torch.cat(branch_feats, dim=2)
                feats.append(f)
            else:
                f = rearrange(out, "b (f h w) c -> b f c h w", f=F_tok, h=H_tok, w=W_tok)
                feats.append(f)

        if self.use_all_layer:
            feats = torch.cat(feats, dim=2)
        else:
            feats = feats[-1]

        return feats.to(torch.float32)

    @torch.no_grad()
    def _build_rynnworld4d_latents(self, pixel_values: torch.Tensor, depth_cond: Optional[torch.Tensor] = None):
        """Build 3 separate latent volumes matching RynnWorld4D inference.

        Args:
            pixel_values: (B, F, C, H, W) in [-1, 1]; frame 0 is the condition.
            depth_cond: optional precomputed frame-0 depth image in [-1, 1],
                shape (B, 1, 3, H, W) or (B, 3, H, W). If None, DA3 estimates it.

        Returns:
            latents_video: frame 0 = VAE(RGB condition), frames 1-20 = noise
            latents_depth: frame 0 = VAE(depth condition), frames 1-20 = noise
            latents_flow:  frame 0 = zero-flow latent (white image), frames 1-20 = noise
        """
        device = self.transformer.device
        B = pixel_values.shape[0]
        condition_frame = pixel_values[:, 0]  # (B, C, H, W) in [-1, 1]

        img_latent = self._vae_encode_single_frame(condition_frame.to(device))
        C_z = img_latent.shape[1]
        H_l, W_l = img_latent.shape[3], img_latent.shape[4]

        vae_temporal_scale = getattr(self.vae.config, 'temporal_compression_ratio', 4)
        num_latent_frames = (self.num_frames - 1) // vae_temporal_scale + 1

        latents_video = torch.randn(B, C_z, num_latent_frames, H_l, W_l, device=device, dtype=self.dtype)
        latents_video[:, :, 0:1, :, :] = img_latent.to(self.dtype)

        if depth_cond is not None:
            depth_frame = depth_cond.to(device)
            if depth_frame.dim() == 5:          # (B, 1, 3, H, W) -> (B, 3, H, W)
                depth_frame = depth_frame[:, 0]
            depth_cond_latent = self._vae_encode_single_frame(depth_frame)
        else:
            depth_cond_latent = self._estimate_depth_latent(condition_frame.to(device))  # DA3 fallback
        latents_depth = torch.randn(B, C_z, num_latent_frames, H_l, W_l, device=device, dtype=self.dtype)
        latents_depth[:, :, 0:1, :, :] = depth_cond_latent.to(self.dtype)

        latents_flow = torch.randn(B, C_z, num_latent_frames, H_l, W_l, device=device, dtype=self.dtype)
        if not hasattr(self, '_zero_flow_latent') or self._zero_flow_latent is None:
            white = torch.ones(1, 3, pixel_values.shape[-2], pixel_values.shape[-1], device=device)
            self._zero_flow_latent = self._vae_encode_single_frame(white).to(self.dtype)
        flow_cond = self._zero_flow_latent.expand(B, -1, -1, -1, -1)
        latents_flow[:, :, 0:1, :, :] = flow_cond

        return latents_video, latents_depth, latents_flow

    @staticmethod
    def _replace_linear_with_int8(module):
        """Recursively swap nn.Linear -> bitsandbytes Linear8bitLt (weight-only int8).

        Returns the number of replaced layers. Quantization itself is deferred
        until the module is moved to CUDA (bitsandbytes quantizes on .cuda()).
        """
        import bitsandbytes as bnb
        n = 0
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                new_layer = bnb.nn.Linear8bitLt(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    has_fp16_weights=False,
                )
                new_layer.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    new_layer.bias.data = child.bias.data.clone()
                setattr(module, name, new_layer)
                n += 1
            else:
                n += WanFeatureExtractor._replace_linear_with_int8(child)
        return n

    def _get_da3_model(self):
        """Lazy-load Depth-Anything-3 model.

        Two modes:
          * da3_quantize=False (default): keep model on CPU; the depth path
            temporarily moves it to GPU then offloads it back (memory offload).
          * da3_quantize=True: replace nn.Linear with bitsandbytes int8 layers
            and move the model to CUDA once. The ~1.7GB int8 model stays
            resident on GPU (no per-call CPU<->GPU transfer).
        """
        if self._da3_model is not None:
            return self._da3_model
        _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        da3_src = os.path.join(_project_root, "third_party", "Depth-Anything-3", "src")
        if not os.path.isdir(os.path.join(da3_src, "depth_anything_3")):
            raise RuntimeError(
                f"Bundled Depth-Anything-3 not found at {da3_src}. "
                "Please clone the repo with submodules or install Depth-Anything-3 manually."
            )
        if da3_src not in sys.path:
            sys.path.insert(0, da3_src)
        from depth_anything_3.api import DepthAnything3
        da3_weights = os.path.join(_project_root, "assets", "da3")
        if not os.path.exists(os.path.join(da3_weights, "model.safetensors")):
            da3_weights = self.depth_model_path
        print(f"[WanFeatureExtractor] Loading DA3 from {da3_weights}")
        model = DepthAnything3.from_pretrained(da3_weights)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        self._da3_quantized = False
        if self.da3_quantize:
            n = self._replace_linear_with_int8(model)
            target = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(target)  # bitsandbytes int8-quantizes Linear weights here
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            self._da3_quantized = True
            print(f"[WanFeatureExtractor] DA3 int8-quantized ({n} Linear layers), resident on {target}")

        self._da3_model = model
        return self._da3_model

    @torch.no_grad()
    def _estimate_depth_latent(self, image: torch.Tensor) -> torch.Tensor:
        """Estimate depth from RGB and VAE-encode it as the depth branch condition.

        DA3 is temporarily moved to GPU for inference then offloaded back to CPU,
        so it never coexists with the RynnWorld4D transformer weights on GPU.

        Args:
            image: (B, 3, H, W) in [-1, 1] (same convention as VAE input).
        Returns:
            latent: (B, C_z, 1, H_l, W_l) normalized depth latent.
        """
        import numpy as np
        from PIL import Image as PILImage

        model = self._get_da3_model()
        device = image.device

        # Move DA3 to GPU temporarily (only in non-quantized offload mode).
        if not self._da3_quantized:
            model.to(device)

        rgb01 = ((image + 1.0) / 2.0).clamp(0.0, 1.0)
        depth_imgs = []
        D_MAX = 5.0
        for i in range(rgb01.shape[0]):
            arr = (rgb01[i].permute(1, 2, 0).float().cpu().numpy() * 255.0).astype(np.uint8)
            pil = PILImage.fromarray(arr)
            prediction = model.inference(
                image=[pil],
                process_res=392,
                process_res_method="upper_bound_resize",
            )
            depth = prediction.depth[0]
            if not isinstance(depth, np.ndarray):
                depth = depth.cpu().numpy()
            import cv2
            depth_resized = cv2.resize(depth, (image.shape[-1], image.shape[-2]), interpolation=cv2.INTER_LINEAR)
            depth_clipped = np.clip(depth_resized, 0.0, D_MAX)
            depth_uint8 = (depth_clipped / D_MAX * 255).astype(np.uint8)
            depth_rgb = np.stack([depth_uint8] * 3, axis=0).astype(np.float32) / 255.0
            depth_tensor = torch.from_numpy(depth_rgb) * 2.0 - 1.0
            depth_imgs.append(depth_tensor)

        # Offload DA3 back to CPU to free GPU memory (non-quantized mode only).
        if not self._da3_quantized:
            model.to("cpu")
            torch.cuda.empty_cache()

        depth_batch = torch.stack(depth_imgs, dim=0).to(device=device)
        return self._vae_encode_single_frame(depth_batch)
