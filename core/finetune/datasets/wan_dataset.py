import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override
import PIL
import json
from core.finetune.constants import LOG_LEVEL, LOG_NAME
import torch.nn.functional as F
from .utils import (
    preprocess_image_with_resize,
    preprocess_video_with_resize,
)
import random
import time
import os
from termcolor import cprint

if TYPE_CHECKING:
    from core.finetune.trainer import Trainer

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(LOG_NAME, LOG_LEVEL)


class RynnWorld4DDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        cache_dir: str,
        device: torch.device,
        trainer: "Trainer" = None,
        prompt: str = "",
    ) -> None:
        super().__init__()
        data_root = Path(data_root)
        if not data_root.exists():
            raise FileNotFoundError(f"Data root not found at: {data_root}")
        with open(data_root, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Filter out items missing required keys
        valid_data = [item for item in data if 'rgb_latents' in item and 'flow_depth_latents' in item]
        if len(valid_data) < len(data):
            cprint(f"⚠️  Filtered out {len(data) - len(valid_data)} items missing 'flow_depth_latents' key", 'yellow')
        self.rgb_latents = [item['rgb_latents'] for item in valid_data]
        self.flow_depth_latents = [item['flow_depth_latents'] for item in valid_data]

        cprint(f"✅  Number of slice: {len(self.rgb_latents)}", 'green')

        self.trainer = trainer
        self.device = device

        if self.trainer is None:
            raise ValueError("A `trainer` object with `encode_video` and `encode_prompt` methods must be provided.")
        self.encode_video = trainer.encode_video
        self.encode_prompt = trainer.encode_prompt
        cache_dir = Path(cache_dir)

        null_prompt = prompt
        self.null_prompt_embedding = self._prepare_null_embedding(null_prompt, cache_dir)

    def __len__(self) -> int:
        return len(self.rgb_latents)

    def _prepare_null_embedding(self, prompt: str, cache_dir: str) -> torch.Tensor:
        cache_dir = Path(cache_dir) / "prompt_embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        embedding_path = cache_dir / f"null_{prompt_hash}.safetensors"
        
        rank = int(os.environ.get('RANK', '0'))
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        
        if embedding_path.exists():
            return load_file(embedding_path)["null_prompt_embedding"]
        
        # Only rank 0 generates the embedding to avoid write race condition
        if local_rank == 0:
            if self.trainer is None or not hasattr(self.trainer, 'encode_prompt'):
                cprint("⚠️ Warning: No trainer provided to encode null_prompt. Using dummy.", 'yellow')
                return torch.zeros(1, 1024)
            
            with torch.no_grad():
                cprint(f"[Rank {rank}] Generating null prompt embedding...", 'cyan')
                emb = self.trainer.encode_prompt(prompt, device=self.device).cpu().squeeze(0) 
                save_file({"null_prompt_embedding": emb}, embedding_path)
                return emb
        else:
            # Wait for rank 0 to write the cache file
            for _ in range(60):
                if embedding_path.exists():
                    return load_file(embedding_path)["null_prompt_embedding"]
                time.sleep(1)
            raise RuntimeError(f"[Rank {rank}] Timeout waiting for null prompt embedding at {embedding_path}")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        video_latent_path = self.rgb_latents[index]
        flow_depth_latent_path = self.flow_depth_latents[index]
        video_latent_path = Path(video_latent_path)
        # frame = self.frame[index]
        max_retries = 5
        cache_data = None

        for i in range(max_retries):
            try:
                cache_data = load_file(video_latent_path)
                flow_depth_cache_data = load_file(flow_depth_latent_path)
                break  
            except Exception as e:
                rank = os.environ.get('RANK', '0')
                if i < max_retries - 1:
                    print(f"[Rank {rank}] Warning: Load failed (attempt {i+1}/{max_retries}). Retrying... Path: {video_latent_path}")
                    time.sleep(1)
                else:
                    print(f"[Rank {rank}] ERROR: Permanent failure on {video_latent_path}. Skipping to random sample.")
                    return self.__getitem__(random.randint(0, len(self) - 1))
        try:
            encoded_video = cache_data["video_latents"]       # [C, T, H, W]
            text_embedding = cache_data["text_embeds"].squeeze(0)  # Remove batch dim: (1, seq, dim) -> (seq, dim)

            encoded_depth = flow_depth_cache_data["depth_latents"]  # [C, T, H, W]
            encoded_flow = flow_depth_cache_data["flow_latents"]    # [C, T, H, W]

            # Shape consistency check: all branches must have the same spatial-temporal dimensions
            if encoded_depth.shape != encoded_video.shape or encoded_flow.shape != encoded_video.shape:
                rank = os.environ.get('RANK', '0')
                print(f"[Rank {rank}] Shape mismatch at index {index}: video={encoded_video.shape}, depth={encoded_depth.shape}, flow={encoded_flow.shape}. Skipping.")
                return self.__getitem__(random.randint(0, len(self) - 1))

            img_latent = encoded_video[:, :1, :, :]       # [C, 1, H, W]
            depth_latent = encoded_depth[:, :1, :, :]     # [C, 1, H, W]
            flow_latent = encoded_flow[:, :1, :, :]       # [C, 1, H, W]

        except Exception as e:
            print(f"Error parsing keys in {video_latent_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        return {
            "encoded_video": encoded_video,
            "encoded_depth": encoded_depth,
            "encoded_flow": encoded_flow,
            "img_latent": img_latent,
            "depth_latent": depth_latent,
            "flow_latent": flow_latent,
            "null_embedding": self.null_prompt_embedding,
            "text_embedding": text_embedding,
        }
