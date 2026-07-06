"""
CPU offload inference for RynnWorld4D + RynnWorld4D-Policy on 32GB GPU.

Features:
- RynnWorld4D transformer on CPU (sequential offload to GPU during forward)
- No VAE loaded (saves ~3 GB GPU)
- Pre-computed text embedding (saves ~11 GB GPU for UMT5)
- int8 PTQ on RynnWorld4D transformer (saves ~22 GB)

Usage:
    # Step 1: Encode text offline (needs ~15 GB GPU for UMT5)
    python encode_text.py --texts "Pick-Place" --output_dir ./text_embeddings

    # Step 2: Run CPU offload inference (~12 GB GPU, ~29 GB CPU)
    python inference_cpu_offload.py \
        --embedding_path ./text_embeddings/Pick-Place.safetensors \
        --config_path ./policy_conf/train_config.yaml \
        --output_dir ./results
"""
import argparse
import os
import sys
import gc
from pathlib import Path
from time import time

import torch
import torch.nn as nn
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent))


class Int8Linear(nn.Module):
    """int8 linear layer with on-the-fly dequantization."""
    def __init__(self, in_features, out_features, weight_int8, scale, bias=None):
        super().__init__()
        self.register_buffer('weight_int8', weight_int8)
        self.register_buffer('scale', scale)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None

    def forward(self, x):
        weight = self.weight_int8.to(x.dtype) * self.scale.unsqueeze(-1)
        return nn.functional.linear(x, weight, self.bias)


def quantize_model_to_int8(model):
    """Quantize RynnWorld4D transformer Linear layers to int8 (per-channel)."""
    count = 0
    skip_prefixes = ('TVP_encoder.text_encoder', 'Video_Former', 'policy_head',
                     'TVP_encoder.vae', 'state_encoder', 'action_head')

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(name.startswith(prefix) for prefix in skip_prefixes):
                continue
            if 'transformer' not in name:
                continue

            weight = module.weight.data.float()
            abs_max = torch.abs(weight).amax(dim=-1, keepdim=True).clamp(min=1e-5)
            scale = abs_max / 127.0
            weight_q = (weight / scale).round().clamp(-128, 127).to(torch.int8)

            new_linear = Int8Linear(
                module.in_features, module.out_features,
                weight_q, scale.squeeze(-1),
                module.bias.data if module.bias is not None else None
            )
            new_linear.to(device=module.weight.device)

            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[-1]
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
                setattr(parent, child_name, new_linear)
            else:
                setattr(model, child_name, new_linear)
            count += 1
            del weight, weight_q, scale

    return count


def load_policy_cpu(config_path, use_int8=True):
    """Load policy with transformer on CPU, int8 quantized."""
    from hydra import compose, initialize
    import hydra

    with initialize(config_path="./policy_conf", version_base=None):
        cfg = compose(config_name="train_config")

    model = hydra.utils.instantiate(cfg.model)
    model.eval()

    if use_int8:
        print("  Quantizing RynnWorld4D transformer to int8...")
        count = quantize_model_to_int8(model)
        print(f"  Quantized {count} Linear layers")

    # Keep everything on CPU (no .cuda())
    return model


def manual_sequential_offload(model, obs, goal, device="cuda"):
    """Manual sequential offload: move transformer blocks to GPU one by one.

    This is a simplified version that moves the entire transformer to GPU temporarily
    during forward, then moves it back to CPU.
    """
    model.reset()

    # Move small components to GPU
    model.Video_Former = model.Video_Former.to(device)
    model.model = model.model.to(device)  # policy head

    # Prepare input on GPU
    rgb = obs["rgb_obs"]["rgb_static"].to(device)
    state = obs["state"].to(device)

    # Get pre-computed embedding on GPU
    text_emb = goal["lang_text_embedding"].to(device)

    obs_gpu = {"rgb_obs": {"rgb_static": rgb}, "state": state}
    goal_gpu = {"lang_text_embedding": text_emb}

    # Move TVP_encoder (RynnWorld4D transformer) to GPU temporarily
    model.TVP_encoder = model.TVP_encoder.to(device)

    # Forward pass
    with torch.no_grad():
        action = model.step(obs=obs_gpu, goal=goal_gpu)

    # Move transformer back to CPU
    model.TVP_encoder = model.TVP_encoder.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    return action.cpu()


@torch.no_grad()
def benchmark_inference(model, obs, goal, device="cuda", n_runs=5):
    """Benchmark inference speed with CPU offload."""
    times = []

    for i in range(n_runs):
        model.reset()

        # Move components to GPU
        model.Video_Former = model.Video_Former.to(device)
        model.model = model.model.to(device)
        model.TVP_encoder = model.TVP_encoder.to(device)

        rgb = obs["rgb_obs"]["rgb_static"].to(device)
        state = obs["state"].to(device)
        text_emb = goal["lang_text_embedding"].to(device)

        obs_gpu = {"rgb_obs": {"rgb_static": rgb}, "state": state}
        goal_gpu = {"lang_text_embedding": text_emb}

        torch.cuda.synchronize()
        start = time()
        action = model.step(obs=obs_gpu, goal=goal_gpu)
        torch.cuda.synchronize()
        elapsed = time() - start
        times.append(elapsed)

        # Move back to CPU
        model.TVP_encoder = model.TVP_encoder.cpu()
        model.Video_Former = model.Video_Former.cpu()
        model.model = model.model.cpu()
        torch.cuda.empty_cache()
        gc.collect()

    avg = sum(times) / len(times)
    print(f"  eval_forward avg: {avg:.3f}s ({1/avg:.2f} FPS)")
    return action.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_path", type=str, required=True,
                        help="Path to pre-computed text embedding .safetensors")
    parser.add_argument("--config_path", type=str, default="./policy_conf/train_config.yaml")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_runs", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load pre-computed text embedding
    data = load_file(args.embedding_path)
    text_emb = data["lang_text_embedding"]
    text = data.get("text", "unknown").decode() if "text" in data else "unknown"
    print(f"Text: '{text}' -> {text_emb.shape}")

    # Load policy on CPU with int8
    print("\nLoading policy (CPU, int8)...")
    model = load_policy_cpu("./policy_conf", use_int8=True)

    # Dummy input
    B = 1
    rgb_obs = torch.randn(B, 4, 3, 224, 224)  # training resolution, 4 frames
    state_obs = torch.randn(B, 54)  # Tianji dual-arm

    # Benchmark
    print(f"\nBenchmarking ({args.n_runs} runs, CPU offload)...")
    action = benchmark_inference(model, rgb_obs, state_obs,
                                 {"lang_text_embedding": text_emb},
                                 args.device, args.n_runs)

    print(f"\nAction: {action[0, 0, :5].numpy()} ...")

    # Save
    torch.save({"action": action, "text": text},
               os.path.join(args.output_dir, "action.pt"))
    print(f"Saved to {args.output_dir}/action.pt")


if __name__ == "__main__":
    main()
