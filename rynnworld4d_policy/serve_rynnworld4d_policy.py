"""
Real-robot deployment server for RynnWorld4D Policy.

Lightweight adapter (Path B): wraps the trained VPP_Policy behind the OpenPI
`BasePolicy.infer(obs) -> {"actions": ...}` interface and serves it with the
same `WebsocketPolicyServer` used by serve_policy.py. The robot client
(tianjiwuji_client_sync_fixed.py) connects unchanged over websocket/msgpack.

Observation contract (from client build_observation):
    "observation/state":            float32 (54,)   raw joint state
    "observation/image":            uint8  (H,W,3)  head camera, RAW (e.g. 720x1280)
    "observation/left_wrist_image": uint8  (H,W,3)  (unused by this policy)
    "observation/right_wrist_image":uint8  (H,W,3)  (unused by this policy)
    "prompt":                       str             (ignored; we use a fixed
                                                     pre-computed UMT5 embedding)

Returned action dict:
    "actions": float32 (action_horizon, 54)   un-normalized joint commands

Image handling: the head image arrives RAW. We replicate the training transform
(CenterCrop to 480x640 -> [0,1] -> normalize to [-1,1]) on the server side, so the
client must NOT pre-resize to 224x224 (see updated client).

Usage:
    python serve_rynnworld4d_policy.py \
        --checkpoint logs/rynnworld4d_fm_tianji_stage2/.../checkpoints/last.pt \
        --port 8000
"""
import argparse
import json
import logging
import os
import socket
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torchvision.transforms as T
from hydra import compose, initialize
from PIL import Image
from safetensors.torch import load_file

# --- make local policy package importable ---
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- make OpenPI server + client packages importable (use bundled copy) ---
OPENPI_ROOT = PROJECT_ROOT / "third_party" / "Openpi_damo"
if not (OPENPI_ROOT / "src").is_dir():
    raise RuntimeError(f"Bundled OpenPI not found at {OPENPI_ROOT}/src; please clone the repo with submodules.")
sys.path.insert(0, str(OPENPI_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "packages" / "openpi-client" / "src"))

import hydra as hydra_lib  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402
from openpi_client import base_policy as _base_policy  # noqa: E402

logger = logging.getLogger(__name__)


class RynnWorld4DPolicy(_base_policy.BasePolicy):
    """OpenPI BasePolicy adapter around the trained VPP_Policy."""

    def __init__(
        self,
        checkpoint: str,
        config_name: str = "train_config",
        text_embedding_path: str = "",
        action_stats_path: str = "",
        device: str = "cuda",
    ):
        self.device = torch.device(device)

        # 1. Build model from the same hydra config used for training.
        with initialize(config_path="./policy_conf", job_name="serve"):
            cfg = compose(config_name=config_name)
        self.cfg = cfg
        self.height = int(cfg.wan_height)   # 480
        self.width = int(cfg.wan_width)     # 640

        logger.info("Instantiating VPP_Policy (loads frozen RynnWorld4D backbone)...")
        model = hydra_lib.utils.instantiate(cfg.model)

        # 2. Load trained weights (optional). The checkpoint stores:
        #    'model' -> full state_dict (incl. frozen 21B backbone),
        #    'ema'   -> EMA of trainable params only (Video_Former + flow head).
        # We overlay EMA trainable params onto the freshly-built model and skip
        # reloading the backbone (it is already loaded from rynnworld4d_ckpt).
        # If no checkpoint is given, the trainable head keeps its random init
        # (useful for deployment-flow / GPU-footprint testing).
        if checkpoint:
            logger.info("Loading checkpoint: %s", checkpoint)
            try:
                ckpt = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
            except (TypeError, RuntimeError):
                ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            ema = ckpt.get("ema", None)
            if ema is not None:
                missing, unexpected = model.load_state_dict(ema, strict=False)
                logger.info(
                    "Loaded EMA params (%d tensors). missing=%d unexpected=%d",
                    len(ema), len(missing), len(unexpected),
                )
            else:
                model.load_state_dict(ckpt["model"], strict=False)
                logger.info("Loaded full model state_dict (no EMA in checkpoint).")
            del ckpt
        else:
            logger.warning(
                "No checkpoint provided: serving with RANDOM-INITIALIZED head "
                "weights. Actions will be meaningless; use only for "
                "deployment-flow / GPU-footprint testing."
            )

        model = model.to(self.device)
        model.process_device()
        model.eval()
        self.model = model

        # 3. Pre-computed UMT5 text embedding (computed offline, saved as safetensors).
        text_embedding_path = text_embedding_path or str(cfg.text_embedding_path)
        data = load_file(text_embedding_path)
        emb = data["lang_text_embedding"]            # (1, 77, 4096)
        if emb.dim() == 3:
            emb = emb.squeeze(0)                      # (77, 4096)
        self.text_embedding = emb.to(self.device)
        logger.info("Loaded text embedding %s from %s",
                    tuple(self.text_embedding.shape), text_embedding_path)

        # 4. Action normalization stats (must match training).
        action_stats_path = action_stats_path or os.path.join(
            str(cfg.root_data_dir), "action_stats.json"
        )
        with open(action_stats_path) as f:
            stats = json.load(f)
        self.action_mean = np.asarray(stats["mean"], dtype=np.float32)
        self.action_std = np.asarray(stats["std"], dtype=np.float32)
        logger.info("Loaded action stats from %s (dim=%d)",
                    action_stats_path, self.action_mean.shape[0])

        # 5. Image transform: replicate dataset eval transform (no augmentation).
        self.transform = T.Compose([
            T.ToPILImage(),
            T.CenterCrop((self.height, self.width)),
            T.ToTensor(),                                   # [0,1]
            T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),     # [-1,1]
        ])

        self.model.reset()
        logger.info("RynnWorld4DPolicy ready.")

    @property
    def metadata(self) -> dict:
        return {
            "policy": "rynnworld4d",
            "action_dim": int(self.action_mean.shape[0]),
            "action_horizon": int(self.cfg.act_seq_len),
            "image_size": [self.height, self.width],
        }

    def _prep_image(self, image: np.ndarray) -> torch.Tensor:
        """Raw uint8 HWC -> (1, 1, 3, H, W) normalized tensor."""
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        t = self.transform(image)                      # (3, H, W)
        return t.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, 3, H, W)

    @torch.no_grad()
    def infer(self, obs: Dict) -> Dict:
        head = np.asarray(obs["observation/image"])
        state = np.asarray(obs["observation/state"], dtype=np.float32)

        rgb_static = self._prep_image(head)
        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)  # (1, 54)

        model_obs = {"rgb_obs": {"rgb_static": rgb_static}, "state": state_t}
        goal = {"lang_text_embedding": self.text_embedding}

        action_pred = self.model.eval_forward(model_obs, goal)  # (1, horizon, 54)
        action_pred = action_pred[0].float().cpu().numpy()       # (horizon, 54)

        # Un-normalize: training normalized actions as (a - mean) / std.
        actions = action_pred * self.action_std + self.action_mean
        return {"actions": actions.astype(np.float32)}

    def reset(self) -> None:
        self.model.reset()


def main():
    parser = argparse.ArgumentParser(description="RynnWorld4D Policy deployment server")
    parser.add_argument(
        "--checkpoint", type=str, default="",
        help="Path to trained .pt checkpoint (contains 'ema' trainable params). "
             "If omitted, serves with random-init head weights (footprint test only).",
    )
    parser.add_argument("--config", type=str, default="train_config")
    parser.add_argument("--text_embedding_path", type=str, default="")
    parser.add_argument("--action_stats_path", type=str, default="")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )

    policy = RynnWorld4DPolicy(
        checkpoint=args.checkpoint,
        config_name=args.config,
        text_embedding_path=args.text_embedding_path,
        action_stats_path=args.action_stats_path,
    )

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        local_ip = "?"
    logger.info("Creating server (host=%s ip=%s port=%d)", hostname, local_ip, args.port)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
