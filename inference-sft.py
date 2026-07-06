"""
inference-sft.py  —  RynnWorld4D SFT checkpoint inference

Modified from: https://github.com/huggingface/diffusers (WanImageToVideoPipeline)
The denoising loop and scheduler interface are adapted from diffusers;
extended with 3-branch (video + depth + flow) joint denoising.

Purpose:
    Load a stage1 SFT (full parameter fine-tuning) checkpoint and run inference.
    Fusion layers are zeroed out since stage1 doesn't train cross-branch communication.

Usage:
    python inference-sft.py \
        --model_path ./pretrained/Wan2.2-TI2V-5B-Diffusers \
        --checkpoint_path training/rynnworld4d-stage1-sft/checkpoint-300 \
        --json_path data/rdt.json \
        --output_dir results/sft-inference \
        --max_samples 4
"""

import argparse
import gc
import json
import os
import shutil
import tempfile
from copy import deepcopy
from typing import Optional

import torch
from safetensors.torch import load_file
from termcolor import cprint

from diffusers import WanImageToVideoPipeline
from diffusers.utils import BaseOutput, export_to_video

try:
    from diffusers.models.transformers.transformer_wan import WanTimeTextImageEmbedding
except ImportError as e:
    raise e

from core.finetune.models.wan_i2v.module import (
    RynnWorld4DTransformer3DModel,
    patched_wan_time_text_image_embedding_forward,
)
from core.finetune.models.wan_i2v.module_joint import JointRynnWorld4DTransformer3DModel

WanTimeTextImageEmbedding.forward = patched_wan_time_text_image_embedding_forward


def randn_tensor(shape, generator=None, device=None, dtype=None):
    if isinstance(device, str):
        device = torch.device(device)
    rand_device = device or torch.device("cpu")
    if generator is not None:
        gen_dev = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_dev != rand_device.type and gen_dev == "cpu":
            rand_device = "cpu"
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]
    latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype)
    if rand_device != device:
        latents = latents.to(device)
    return latents


def zero_fusion_layers(transformer):
    count = 0
    for block in transformer.blocks:
        for name in [
            "video_to_depth_zero", "video_to_flow_zero",
            "depth_to_video_zero", "flow_to_video_zero",
        ]:
            layer = getattr(block, name, None)
            if layer is not None:
                layer.weight.data.zero_()
                layer.bias.data.zero_()
                count += 1
    cprint(f"Zeroed {count} fusion layers — branches are fully independent.", "yellow")


def load_sft_checkpoint(checkpoint_path, transformer):
    """Load SFT checkpoint incrementally to reduce peak CPU memory."""
    model_states_path = os.path.join(checkpoint_path, "pytorch_model", "mp_rank_00_model_states.pt")
    if not os.path.exists(model_states_path):
        raise FileNotFoundError(f"SFT model states not found at {model_states_path}")
    cprint(f"Loading SFT checkpoint from {model_states_path} ...", "cyan")

    import hashlib
    cache_name = "sft_" + hashlib.md5(model_states_path.encode()).hexdigest() + ".pt"
    local_path = os.path.join("/tmp", cache_name)
    if not os.path.exists(local_path):
        import shutil
        cprint("  Copying to local disk for fast loading ...", "cyan")
        shutil.copy2(model_states_path, local_path)
        cprint("  Copy complete.", "cyan")
    else:
        cprint("  Found local copy, using it.", "cyan")

    checkpoint = torch.load(local_path, map_location="cpu", weights_only=False, mmap=True)
    state_dict = checkpoint["module"]
    cprint(f"Checkpoint has {len(state_dict)} keys.", "cyan")

    # Rename keys in-place
    keys_to_rename = [k for k in state_dict if "module." in k or ".base_layer." in k]
    for k in keys_to_rename:
        clean_key = k.replace("module.", "").replace(".base_layer.", ".")
        if clean_key != k:
            v = state_dict.pop(k)
            state_dict[clean_key] = v

    # Load parameters incrementally to avoid doubling memory
    model_sd = transformer.state_dict()
    loaded, missing_keys = 0, []
    for k in list(state_dict.keys()):
        if k in model_sd:
            model_sd[k].copy_(state_dict[k])
            loaded += 1
        else:
            missing_keys.append(k)
    cprint(f"SFT checkpoint loaded incrementally. Loaded: {loaded}, Missing: {len(missing_keys)}.", "green")

    del model_sd, checkpoint
    state_dict.clear()
    del state_dict
    import gc
    gc.collect()
    torch.cuda.empty_cache()


class RynnWorld4DInferencePipeline(WanImageToVideoPipeline):
    @torch.no_grad()
    def __call__(
        self,
        img_latent: torch.Tensor,
        depth_latent_cond: torch.Tensor,
        flow_latent_cond: torch.Tensor,
        prompt_embeds: torch.Tensor,
        video_latent_gt: torch.Tensor,
        depth_latent_gt: torch.Tensor = None,
        flow_latent_gt: torch.Tensor = None,
        prompt: str = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        output_type: str = "np",
    ):
        device = self._execution_device
        transformer_dtype = self.transformer.dtype

        _, num_channels, _, latent_h, latent_w = img_latent.shape
        num_latent_frames = video_latent_gt.shape[1]

        shape = (1, num_channels, num_latent_frames, latent_h, latent_w)
        # Match training: all three branches start from the SAME noise so denoising
        # trajectories are aligned. See rynnworld4d_trainer.py:802-804
        # (noise_depth = noise_video.clone(); noise_flow = noise_video.clone()).
        noise = randn_tensor(shape, generator=generator, device=device, dtype=transformer_dtype)
        latents_video = noise.clone()
        latents_depth = noise.clone()
        latents_flow  = noise.clone()
        del noise

        first_frame_mask = torch.ones(
            1, 1, num_latent_frames, latent_h, latent_w,
            device=device, dtype=transformer_dtype,
        )
        first_frame_mask[:, :, 0] = 0

        self._guidance_scale = guidance_scale

        prompt_embeds     = prompt_embeds.to(device=device, dtype=transformer_dtype)
        img_latent        = img_latent.to(device=device, dtype=transformer_dtype)
        depth_latent_cond = depth_latent_cond.to(device=device, dtype=transformer_dtype)
        flow_latent_cond  = flow_latent_cond.to(device=device, dtype=transformer_dtype)

        do_cfg = guidance_scale > 1.0
        null_embeds = None
        if do_cfg:
            null_path = os.environ.get(
                "RYNNWORLD4D_NULL_PROMPT",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "null_prompt_embedding.safetensors"),
            )
            from safetensors.torch import load_file as _load_file
            null_embeds = _load_file(null_path)["null_prompt_embedding"].unsqueeze(0).to(device=device, dtype=transformer_dtype)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        depth_scheduler = deepcopy(self.scheduler)
        flow_scheduler  = deepcopy(self.scheduler)
        depth_scheduler.set_timesteps(num_inference_steps, device=device)
        flow_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Precompute the per-token timestep mask once: shape (1, num_tokens) with 0 at first
        # frame and 1 elsewhere. The per-step value is just this * t, no need to redo ::2
        # indexing or .flatten() inside the loop.
        ts_mask = first_frame_mask[0][0][:, ::2, ::2].flatten().unsqueeze(0)  # (1, num_tokens)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                self._current_timestep = t

                latents_video[:, :, 0:1, :, :] = img_latent
                latents_depth[:, :, 0:1, :, :] = depth_latent_cond
                latents_flow[:, :, 0:1, :, :]  = flow_latent_cond

                timestep = ts_mask * t

                video_pred, depth_pred, flow_pred = self.transformer(
                    hidden_states=latents_video,
                    hidden_states_depth=latents_depth,
                    hidden_states_flow=latents_flow,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=None,
                    return_dict=False,
                )

                if do_cfg:
                    video_pred_uncond, depth_pred_uncond, flow_pred_uncond = self.transformer(
                        hidden_states=latents_video,
                        hidden_states_depth=latents_depth,
                        hidden_states_flow=latents_flow,
                        timestep=timestep,
                        encoder_hidden_states=null_embeds,
                        encoder_hidden_states_image=None,
                        return_dict=False,
                    )
                    video_pred = video_pred_uncond + guidance_scale * (video_pred - video_pred_uncond)
                    depth_pred = depth_pred_uncond + guidance_scale * (depth_pred - depth_pred_uncond)
                    flow_pred  = flow_pred_uncond  + guidance_scale * (flow_pred  - flow_pred_uncond)

                latents_video = self.scheduler.step(video_pred, t, latents_video, return_dict=False)[0]
                latents_depth = depth_scheduler.step(depth_pred, t, latents_depth, return_dict=False)[0]
                latents_flow  = flow_scheduler.step(flow_pred, t, latents_flow, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None

        latents_video[:, :, 0:1, :, :] = img_latent
        latents_depth[:, :, 0:1, :, :] = depth_latent_cond
        latents_flow[:, :, 0:1, :, :]  = flow_latent_cond

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(device=device, dtype=self.vae.dtype)
        )
        latents_std = 1.0 / (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(device=device, dtype=self.vae.dtype)
        )

        def decode(lat):
            lat = lat.to(self.vae.dtype)
            lat = lat / latents_std + latents_mean
            vid = self.vae.decode(lat, return_dict=False)[0]
            return self.video_processor.postprocess_video(vid, output_type=output_type)

        video_out = decode(latents_video)
        del latents_video; torch.cuda.empty_cache()
        depth_out = decode(latents_depth)
        del latents_depth; torch.cuda.empty_cache()
        flow_out  = decode(latents_flow)
        del latents_flow; torch.cuda.empty_cache()
        gt_out    = decode(video_latent_gt.unsqueeze(0).to(device=device))
        torch.cuda.empty_cache()
        gt_depth_out = decode(depth_latent_gt.unsqueeze(0).to(device=device)) if depth_latent_gt is not None else None
        torch.cuda.empty_cache()
        gt_flow_out = decode(flow_latent_gt.unsqueeze(0).to(device=device)) if flow_latent_gt is not None else None
        torch.cuda.empty_cache()

        self.maybe_free_model_hooks()
        return (
            BaseOutput(frames=video_out),
            BaseOutput(frames=gt_out),
            BaseOutput(frames=depth_out),
            BaseOutput(frames=flow_out),
            BaseOutput(frames=gt_depth_out) if gt_depth_out is not None else None,
            BaseOutput(frames=gt_flow_out) if gt_flow_out is not None else None,
        )


def main():
    # Disable torch.compile workers to reduce memory usage
    os.environ["TORCHINDUCTOR_MAX_WORKERS"] = "1"

    parser = argparse.ArgumentParser(description="inference-sft: SFT checkpoint inference (fusion zeroed)")
    parser.add_argument("--model_path", type=str, default="./pretrained/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--checkpoint_path", type=str, default="./training/rynnworld4d-stage2-joint-attention/checkpoint-2000")
    parser.add_argument("--json_path", type=str, default="./data/sample.json")
    parser.add_argument("--output_dir", type=str, default="./results/inference-sft")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=40, help="(Legacy) total cap; ignored when --per_dataset_samples > 0.")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--per_dataset_samples", type=int, default=5, help="If > 0, take this many samples from each dataset listed in DEFAULT_DATASET_JSONS instead of using a single hardcoded JSON.")
    parser.add_argument("--share_ffn", type=lambda x: (str(x).lower() == 'true'), default=False, help="Must match training config. Set False for independent-ffn checkpoints.")
    parser.add_argument("--fusion_mode", type=str, default="joint", choices=["none", "unidirectional", "bidirectional", "joint"],
                        help="Must match training config. stage2-new.sh and stage2-rope.sh use 'joint'.")
    parser.add_argument("--joint_start_layer", type=int, default=0, help="Must match training config. stage2-new.sh and stage2-rope.sh both use 0.")
    parser.add_argument("--joint_end_layer", type=int, default=20, help="Must match training config. stage2-new.sh and stage2-rope.sh both use 20. -1 means num_layers.")
    parser.add_argument("--joint_every_n_layers", type=int, default=2, help="Must match training config. stage2-new.sh and stage2-rope.sh both use 2.")
    parser.add_argument("--joint_frame_wise", type=lambda x: (str(x).lower() == 'true'), default=True, help="Must match training config. If True, joint cross-modal attention is restricted to same-frame tokens. Both stage2-new.sh and stage2-rope.sh set this True.")
    parser.add_argument("--joint_use_rope", type=lambda x: (str(x).lower() == 'true'), default=False, help="Must match training config. If True, apply 3D RoPE to Q/K in joint cross-modal attention. stage2-rope.sh sets this True; stage2-new.sh leaves it False.")
    parser.add_argument("--disable_joint", type=lambda x: (str(x).lower() == 'true'), default=False, help="If True, disable joint attention at inference (ablation). Architecture is still built as fusion_mode=joint so the same checkpoint can be loaded; only forward path is bypassed.")
    parser.add_argument("--use_ema", type=lambda x: (str(x).lower() == 'true'), default=False, help="Whether to load EMA weights instead of raw training weights.")
    parser.add_argument("--zero_fusion", type=lambda x: (str(x).lower() == 'true'), default=False, help="Whether to zero out fusion layers. Default False for stage2 (joint attention is trained); set True only for stage1 SFT checkpoints where fusion was untrained.")
    parser.add_argument("--sample_indices", type=str, default=None, help="Comma-separated indices to run (e.g. '3,22,23,24'). If not set, runs all.")
    parser.add_argument("--keep_on_gpu", type=lambda x: (str(x).lower() == 'true'), default=False, help="If True, keep the entire pipeline on GPU (no model_cpu_offload). Faster across multiple samples but needs ~30GB+ VRAM.")
    parser.add_argument("--vae_tiling", type=lambda x: (str(x).lower() == 'true'), default=False, help="Enable VAE tiled decode. Default OFF: diffusers 0.35.2 WanVAE.tiled_decode has a shape-mismatch bug in avg_shortcut (temporal dim 2 vs 4). Non-tiled path decodes all frames together and is robust. Set True only if you confirmed your env doesn't hit the bug.")
    args = parser.parse_args()

    if args.disable_joint and args.fusion_mode != "joint":
        cprint("WARNING: --disable_joint is only meaningful when --fusion_mode=joint; ignoring.", "yellow")
        args.disable_joint = False

    if args.disable_joint:
        args.output_dir = args.output_dir.rstrip("/") + "_joint-off"
        cprint(f"Joint attention DISABLED at inference. Output dir suffixed: {args.output_dir}", "yellow")
    elif args.fusion_mode == "joint":
        args.output_dir = args.output_dir.rstrip("/") + "_joint-on"
        cprint(f"Joint attention ENABLED at inference. Output dir suffixed: {args.output_dir}", "green")

    os.makedirs(args.output_dir, exist_ok=True)

    DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    DEFAULT_DATASET_JSONS = [
        ("agibot",      f"{DATA_ROOT}/agibot.json"),
        ("egovid",      f"{DATA_ROOT}/egovid.json"),
        ("epic",        f"{DATA_ROOT}/epic.json"),
        ("galaxea",     f"{DATA_ROOT}/galaxea.json"),
        ("robocoin",    f"{DATA_ROOT}/robocoin.json"),
        ("robomind",    f"{DATA_ROOT}/robomind.json"),
        ("rdt",         f"{DATA_ROOT}/rdt.json"),
        ("tianji_wuji", f"{DATA_ROOT}/tianji_wuji.json"),
    ]

    DATASET_SAMPLE_INDICES = {
        # galaxea early indices are dominated by repetitive thermostat button-pressing.
        # Override with diverse manipulation scenes (lemon, red fruit, cushion, box, bed covers).
        "galaxea": [500, 1000, 10000, 100000, 150000],
    }

    data = []
    if args.per_dataset_samples and args.per_dataset_samples > 0:
        for ds_name, jp in DEFAULT_DATASET_JSONS:
            if not os.path.exists(jp):
                cprint(f"⚠️ JSON not found, skipping: {jp}", "yellow")
                continue
            with open(jp, "r", encoding="utf-8") as f:
                items = json.load(f)
            items = [item for item in items if 'rgb_latents' in item and 'flow_depth_latents' in item]
            if ds_name in DATASET_SAMPLE_INDICES:
                idx_list = DATASET_SAMPLE_INDICES[ds_name][:args.per_dataset_samples]
                selected = [items[i] for i in idx_list if i < len(items)]
                cprint(f"  {ds_name}: selected {len(selected)} samples by explicit indices {idx_list}", "cyan")
            else:
                selected = items[args.start_idx:][:args.per_dataset_samples]
                cprint(f"  {ds_name}: selected {len(selected)} / {len(items)} valid samples", "cyan")
            for item in selected:
                item["__dataset__"] = ds_name
            data.extend(selected)
    else:
        jp = args.json_path
        if not os.path.exists(jp):
            cprint(f"⚠️ JSON not found: {jp}", "yellow")
        else:
            with open(jp, "r", encoding="utf-8") as f:
                items = json.load(f)
            items = [item for item in items if 'rgb_latents' in item and 'flow_depth_latents' in item]
            selected = items[args.start_idx:][:args.max_samples]
            for item in selected:
                item["__dataset__"] = os.path.splitext(os.path.basename(jp))[0]
            cprint(f"  {os.path.basename(jp)}: selected {len(selected)} samples", "cyan")
            data.extend(selected)
    cprint(f"Total samples to process: {len(data)}", "cyan")

    cprint("Loading pipeline ...", "cyan")
    if args.fusion_mode == "joint":
        transformer = JointRynnWorld4DTransformer3DModel.from_pretrained(
            args.model_path, subfolder="transformer", torch_dtype=torch.bfloat16, eps=1e-5,
            share_ffn=args.share_ffn,
            joint_start_layer=args.joint_start_layer,
            joint_end_layer=args.joint_end_layer,
            joint_every_n_layers=args.joint_every_n_layers,
            joint_frame_wise=args.joint_frame_wise,
            joint_use_rope=args.joint_use_rope,
            low_cpu_mem_usage=True,
        )
        cprint(
            f"Joint attention config: frame_wise={args.joint_frame_wise}, use_rope={args.joint_use_rope}, "
            f"layers=[{args.joint_start_layer}, {args.joint_end_layer}) every {args.joint_every_n_layers}.",
            "green",
        )
    else:
        transformer = RynnWorld4DTransformer3DModel.from_pretrained(
            args.model_path, subfolder="transformer", torch_dtype=torch.bfloat16, eps=1e-5,
            share_ffn=args.share_ffn, fusion_mode=args.fusion_mode,
            low_cpu_mem_usage=True,
        )

    # Create pipeline first (loads text_encoder + vae) before SFT checkpoint
    # to reduce peak memory: avoid transformer + SFT + text_encoder all in RAM at once
    pipe = RynnWorld4DInferencePipeline.from_pretrained(args.model_path, transformer=transformer, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

    # ═══ Load SFT checkpoint ═══
    load_sft_checkpoint(args.checkpoint_path, pipe.transformer)
    gc.collect()
    torch.cuda.empty_cache()

    # ═══ Load EMA weights (optional) ═══
    if args.use_ema:
        ema_path = os.path.join(args.checkpoint_path, "ema_weights.pt")
        if os.path.exists(ema_path):
            ema_state_dict = torch.load(ema_path, map_location="cpu", weights_only=False)
            model_sd = pipe.transformer.state_dict()
            for name, ema_param in ema_state_dict.items():
                if name in model_sd:
                    model_sd[name] = ema_param
            pipe.transformer.load_state_dict(model_sd, strict=False)
            cprint(f"EMA weights loaded from {ema_path} ({len(ema_state_dict)} params overridden).", "green")
            del ema_state_dict, model_sd
            gc.collect()
        else:
            cprint(f"⚠️ --use_ema specified but {ema_path} not found. Using raw weights.", "yellow")

    cprint(f"Scheduler flow_shift = {pipe.scheduler.config.flow_shift} (flow-shifted sigma).", "green")

    # ═══ ZERO FUSION LAYERS (only for stage1 where fusion is not trained) ═══
    if args.zero_fusion:
        zero_fusion_layers(pipe.transformer)
    else:
        cprint("Skipping fusion layer zeroing — using trained fusion weights.", "green")

    # ═══ Joint attention ablation switch ═══
    # Same architecture / same checkpoint; just bypass the joint cross-modal pathway.
    if args.disable_joint:
        n_off, n_total = 0, 0
        for block in pipe.transformer.blocks:
            if hasattr(block, "enable_joint"):
                n_total += 1
                if block.enable_joint:
                    block.enable_joint = False
                    n_off += 1
        cprint(f"Joint attention disabled on {n_off}/{n_total} blocks (ablation mode).", "yellow")

    # Enable VAE tiling + slicing — VAE decode is the second-largest VRAM hog after the
    # transformer, and tiling makes it run in fixed-size chunks regardless of T/H/W.
    # This also avoids triggering CPU offload movement for the VAE on each decode.
    if args.vae_tiling and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    elif not args.vae_tiling:
        cprint("VAE tiling DISABLED (workaround for diffusers tiled_decode shape bug).", "yellow")
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()

    if args.keep_on_gpu:
        pipe.to("cuda")
        cprint("Pipeline ready (kept entirely on GPU; faster but more VRAM).\n", "green")
    else:
        pipe.enable_model_cpu_offload()
        cprint("Pipeline ready (model CPU offload enabled).\n", "green")

    # Force clean memory state
    torch.cuda.empty_cache()
    gc.collect()

    dtype = torch.bfloat16
    cprint(f"Initial VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB", "blue")

    # Use inference_mode for maximum memory efficiency
    sample_indices = None
    if args.sample_indices:
        sample_indices = set(int(x.strip()) for x in args.sample_indices.split(","))
        cprint(f"Only running sample indices: {sorted(sample_indices)}", "cyan")

    with torch.inference_mode():
        for idx, item in enumerate(data):
            if sample_indices is not None and idx not in sample_indices:
                continue
            prompt_text = item.get("prompt", "")
            ds_name = item.get("__dataset__", "unknown")
            generator = torch.Generator(device="cuda").manual_seed(args.seed + idx)
            cprint(f"[{idx+1}/{len(data)}] [{ds_name}] {prompt_text[:80]}", "cyan")

            rgb_data = load_file(item["rgb_latents"])
            fd_data  = load_file(item["flow_depth_latents"])

            video_latents_gt = rgb_data["video_latents"].to(dtype=dtype)
            depth_latents    = fd_data["depth_latents"].to(dtype=dtype)
            flow_latents     = fd_data["flow_latents"].to(dtype=dtype)
            text_embeds      = rgb_data["text_embeds"].to(dtype=dtype)

            img_latent        = video_latents_gt[:, :1, :, :].unsqueeze(0)
            depth_latent_cond = depth_latents[:, :1, :, :].unsqueeze(0)
            flow_latent_cond  = flow_latents[:, :1, :, :].unsqueeze(0)

            output, gt, depth, flow, gt_depth, gt_flow = pipe(
                img_latent=img_latent,
                depth_latent_cond=depth_latent_cond,
                flow_latent_cond=flow_latent_cond,
                prompt_embeds=text_embeds,
                video_latent_gt=video_latents_gt,
                depth_latent_gt=depth_latents,
                flow_latent_gt=flow_latents,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                save_list = [
                    ("rgb", output.frames[0]),
                    ("gt", gt.frames[0]),
                    ("depth", depth.frames[0]),
                    ("flow", flow.frames[0]),
                ]
                if gt_depth is not None:
                    save_list.append(("gt_depth", gt_depth.frames[0]))
                if gt_flow is not None:
                    save_list.append(("gt_flow", gt_flow.frames[0]))
                for tag, frames in save_list:
                    tmp_p = os.path.join(tmpdir, f"{ds_name}_{idx}_{tag}.mp4")
                    out_p = os.path.join(args.output_dir, f"{ds_name}_{idx}_{tag}.mp4")
                    export_to_video(frames, tmp_p, fps=16)
                    shutil.copy(tmp_p, out_p)
                    cprint(f"  Saved {out_p}", "green")

            gc.collect()
            torch.cuda.empty_cache()
            cprint(f"  VRAM after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB", "blue")

    cprint("\nAll done.", "green")


if __name__ == "__main__":
    main()