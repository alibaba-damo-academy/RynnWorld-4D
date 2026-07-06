"""
Inference with pre-computed text embeddings (saves GPU memory).

Usage:
    # Step 1: Encode text offline
    python encode_text.py --texts "Pick-Place" --output_dir ./text_embeddings

    # Step 2: Run inference with pre-computed embeddings
    CUDA_VISIBLE_DEVICES=0 python inference_with_embeddings.py \
        --embedding_path ./text_embeddings/Pick-Place.safetensors \
        --checkpoint_path /path/to/rynnworld4d-policy-ckpt.pt \
        --output_dir ./results
"""
import argparse
import os
import torch
from safetensors.torch import load_file


def load_policy(checkpoint_path, device="cuda"):
    """Load rynnworld4d-policy checkpoint."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from hydra import compose, initialize
    import hydra

    with initialize(config_path="./policy_conf", version_base=None):
        cfg = compose(config_name="train_config")

    model = hydra.utils.instantiate(cfg.model)
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")

    model = model.to(device).eval()
    return model


@torch.no_grad()
def run_inference(model, rgb_obs, state_obs, text_embedding, device="cuda"):
    """Run policy inference with pre-computed text embedding.

    Args:
        model: VPP_Policy
        rgb_obs: RGB tensor (B, F, 3, H, W)
        state_obs: State tensor (B, state_dim)
        text_embedding: Pre-computed UMT5 embedding (1, seq_len, 4096)
        device: Device to run on

    Returns:
        action: (action_dim,) tensor
    """
    model.reset()

    obs = {
        "rgb_obs": {"rgb_static": rgb_obs.to(device)},
        "state": state_obs.to(device),
    }
    goal = {"lang_text_embedding": text_embedding.to(device)}

    action = model.step(obs=obs, goal=goal)
    return action.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_path", type=str, required=True,
                        help="Path to .safetensors file with pre-computed text embedding")
    parser.add_argument("--checkpoint_path", type=str, default="",
                        help="Path to rynnworld4d-policy .pt checkpoint")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load pre-computed text embedding
    data = load_file(args.embedding_path)
    text_embedding = data["lang_text_embedding"]  # (1, seq_len, 4096)
    text = data.get("text", "unknown").decode() if "text" in data else "unknown"
    print(f"Loaded text embedding: '{text}' -> {text_embedding.shape}")

    # Load policy
    print("Loading rynnworld4d-policy...")
    model = load_policy(args.checkpoint_path, args.device)

    # Dummy input (replace with real data)
    B = 1
    rgb_obs = torch.randn(B, 4, 3, 224, 224)  # 4 frames, training resolution
    state_obs = torch.randn(B, 54)  # Tianji dual-arm state

    # Inference
    print("Running inference...")
    action = run_inference(model, rgb_obs, state_obs, text_embedding, args.device)
    print(f"Action shape: {action.shape}")
    print(f"Action: {action[0, 0, :5].numpy()} ...")

    # Save result
    torch.save({"action": action, "text": text},
               os.path.join(args.output_dir, "action.pt"))
    print(f"Saved action to {args.output_dir}/action.pt")


if __name__ == "__main__":
    main()
