"""
Dataset for Tianji/Wuji robot episodes (mp4 video + parquet actions).

Each episode dir contains:
  observation.images.head.mp4   – head camera video (1280×720, 30fps)
  timeseries.parquet            – per-frame actions & states
  metadata.json                 – task prompt, fps, total_frames

Output format is compatible with VPP_policy.extract_predictive_feature:
  {
    "rgb_obs": {"rgb_static": (1, 3, H, W)},
    "lang_text": ["Pick-Place"],
    "actions": (action_seq_len, action_dim),
    "idx": int,
  }
"""
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from safetensors.torch import load_file


class TianjiVideoDataset(Dataset):
    """Reads mp4 + parquet from Tianji robot episodes.

    Args:
        data_dir: path containing episode_* subdirectories.
        action_col: parquet column name for action vector.
        action_seq_len: number of future action frames per sample.
        height, width: target image size (after resize).
        transform: optional torchvision transform applied to each RGB frame.
        skip_frames: subsample rate (1 = use every frame).
    """

    def __init__(
        self,
        data_dir: str,
        action_col: str = "action",
        action_seq_len: int = 10,
        height: int = 224,
        width: int = 224,
        transform: Optional[Any] = None,
        skip_frames: int = 1,
        normalize_actions: bool = True,
        augment: bool = True,
        obs_seq_len: int = 1,
        text_embedding_path: str = "",
        depth_data_dir: str = "",
        depth_subpath: str = "exports/mini_npz/depth.mp4",
    ):
        super().__init__()
        self.action_col = action_col
        self.action_seq_len = action_seq_len
        self.height = height
        self.width = width
        self.skip_frames = skip_frames
        self.normalize_actions = normalize_actions
        self.obs_seq_len = obs_seq_len
        self.depth_data_dir = depth_data_dir
        self.depth_subpath = depth_subpath

        # Load pre-computed text embedding if provided
        self.text_embedding = None
        if text_embedding_path and os.path.exists(text_embedding_path):
            data = load_file(text_embedding_path)
            self.text_embedding = data["lang_text_embedding"].squeeze(0)  # Remove batch dim: (1, seq, dim) -> (seq, dim)
            print(f"Loaded pre-computed text embedding: {self.text_embedding.shape}")

        if transform is None:
            transforms_list = [
                T.ToPILImage(),
                T.CenterCrop((height, width)),
            ]
            if augment:
                transforms_list.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
            transforms_list += [
                T.ToTensor(),                          # [0, 1]
                T.Normalize(mean=[0.5]*3, std=[0.5]*3), # [-1, 1]
            ]
            self.transform = T.Compose(transforms_list)
        else:
            self.transform = transform

        # Depth transform: same geometric crop + normalization, never augmented.
        self.depth_transform = T.Compose([
            T.ToPILImage(),
            T.CenterCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ])

        # Scan episodes
        episode_dirs = sorted(glob.glob(os.path.join(data_dir, "episode_*")))
        assert len(episode_dirs) > 0, f"No episodes found in {data_dir}"

        self.episodes: List[Dict] = []
        self.sample_index: List[tuple] = []  # (ep_idx, start_frame)

        for ep_idx, ep_dir in enumerate(episode_dirs):
            parquet_path = os.path.join(ep_dir, "timeseries.parquet")
            video_path = os.path.join(ep_dir, "observation.images.head.mp4")
            left_wrist_path = os.path.join(ep_dir, "observation.images.left_wrist.mp4")
            right_wrist_path = os.path.join(ep_dir, "observation.images.right_wrist.mp4")
            meta_path = os.path.join(ep_dir, "metadata.json")

            # Resolve depth video (per-frame depth, aligned with head RGB).
            depth_path = None
            if self.depth_data_dir:
                ep_name = os.path.basename(ep_dir)
                cand = os.path.join(self.depth_data_dir, ep_name, self.depth_subpath)
                if os.path.exists(cand):
                    depth_path = cand

            if not os.path.exists(parquet_path) or not os.path.exists(video_path):
                continue

            df = pd.read_parquet(parquet_path)
            actions = np.stack(df[action_col].values).astype(np.float32)
            states = np.stack(df["observation.state"].values).astype(np.float32)
            n_frames = len(df)

            # Read task prompt from metadata
            task_prompt = "Pick-Place"
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                task_prompt = meta.get("task_prompt", task_prompt)

            self.episodes.append({
                "video_path": video_path,
                "depth_path": depth_path,
                "left_wrist_path": left_wrist_path if os.path.exists(left_wrist_path) else None,
                "right_wrist_path": right_wrist_path if os.path.exists(right_wrist_path) else None,
                "actions": actions,       # (n_frames, action_dim)
                "states": states,         # (n_frames, state_dim)
                "n_frames": n_frames,
                "task_prompt": task_prompt,
            })

            # Build sample index: every valid starting frame
            max_start = n_frames - action_seq_len
            for start in range(0, max_start, skip_frames):
                self.sample_index.append((ep_idx, start))

        self.action_dim = self.episodes[0]["actions"].shape[1]

        # Action normalization: load or compute per-dim mean/std
        self.action_mean = None
        self.action_std = None
        if self.normalize_actions:
            stats_path = os.path.join(data_dir, "action_stats.json")
            if os.path.exists(stats_path):
                with open(stats_path) as f:
                    stats = json.load(f)
                self.action_mean = np.array(stats["mean"], dtype=np.float32)
                self.action_std = np.array(stats["std"], dtype=np.float32)
            else:
                # Compute from loaded episodes
                all_acts = np.concatenate([ep["actions"] for ep in self.episodes])
                self.action_mean = all_acts.mean(axis=0)
                self.action_std = np.maximum(all_acts.std(axis=0), 1e-6)
                # Persist so inference/deployment uses the exact same stats.
                try:
                    with open(stats_path, "w") as f:
                        json.dump(
                            {
                                "mean": self.action_mean.tolist(),
                                "std": self.action_std.tolist(),
                            },
                            f,
                            indent=2,
                        )
                    print(f"Saved action stats to {stats_path}")
                except OSError as e:
                    print(f"WARNING: could not save action stats to {stats_path}: {e}")

        n_with_depth = sum(1 for ep in self.episodes if ep["depth_path"] is not None)
        print(
            f"TianjiVideoDataset: {len(self.episodes)} episodes, "
            f"{len(self.sample_index)} samples, action_dim={self.action_dim}, "
            f"normalize={self.normalize_actions}, with_depth={n_with_depth}/{len(self.episodes)}"
        )

    def __len__(self):
        return len(self.sample_index)

    def _read_frame(self, video_path, frame_idx):
        """Read a single frame from video, return RGB numpy array."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        assert ret, f"Failed to read frame {frame_idx} from {video_path}"
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_idx, start_frame = self.sample_index[idx]
        ep = self.episodes[ep_idx]

        # Read obs_seq_len frames ending at start_frame (inclusive)
        frames = []
        for i in range(self.obs_seq_len):
            frame_idx = max(0, start_frame - (self.obs_seq_len - 1 - i))
            frame = self._read_frame(ep["video_path"], frame_idx)
            frames.append(self.transform(frame))
        rgb_static = torch.stack(frames, dim=0)  # (obs_seq_len, 3, H, W)

        rgb_obs = {"rgb_static": rgb_static}

        # Frame-0 depth condition (only the current frame is needed: RynnWorld4D
        # fills real depth at latent frame 0 only). Falls back to DA3 at the
        # extractor if no depth video is available.
        depth_static = None
        if ep["depth_path"] is not None:
            depth_frame = self._read_frame(ep["depth_path"], start_frame)
            depth_static = self.depth_transform(depth_frame).unsqueeze(0)  # (1, 3, H, W)

        # Read wrist camera frames if available
        if ep["left_wrist_path"] is not None:
            lw_frame = self._read_frame(ep["left_wrist_path"], start_frame)
            rgb_obs["rgb_left_wrist"] = self.transform(lw_frame).unsqueeze(0)
        if ep["right_wrist_path"] is not None:
            rw_frame = self._read_frame(ep["right_wrist_path"], start_frame)
            rgb_obs["rgb_right_wrist"] = self.transform(rw_frame).unsqueeze(0)

        # Current state (proprioception): observation.state at current frame
        current_state = ep["states"][start_frame].copy()
        state = torch.from_numpy(current_state)  # (state_dim,) — raw, no normalization

        # Action sequence: next action_seq_len frames starting from start_frame
        actions = ep["actions"][start_frame : start_frame + self.action_seq_len]
        if self.normalize_actions and self.action_mean is not None:
            actions = (actions - self.action_mean) / self.action_std
        actions = torch.from_numpy(actions)  # (action_seq_len, action_dim)

        result = {
            "rgb_obs": rgb_obs,
            "state": state,
            "actions": actions,
            "idx": idx,
        }

        if depth_static is not None:
            result["depth_static"] = depth_static

        if self.text_embedding is not None:
            result["lang_text_embedding"] = self.text_embedding.clone()
        else:
            result["lang_text"] = [ep["task_prompt"]]

        return result


def collate_tianji(batch: List[Dict]) -> Dict:
    """Custom collate that handles lang_text_embedding (tensor) or lang_text (string) and multi-camera properly."""
    rgb_obs = {"rgb_static": torch.stack([b["rgb_obs"]["rgb_static"] for b in batch])}
    if "rgb_left_wrist" in batch[0]["rgb_obs"]:
        rgb_obs["rgb_left_wrist"] = torch.stack([b["rgb_obs"]["rgb_left_wrist"] for b in batch])
    if "rgb_right_wrist" in batch[0]["rgb_obs"]:
        rgb_obs["rgb_right_wrist"] = torch.stack([b["rgb_obs"]["rgb_right_wrist"] for b in batch])

    result = {
        "rgb_obs": rgb_obs,
        "state": torch.stack([b["state"] for b in batch]),
        "actions": torch.stack([b["actions"] for b in batch]),
        "idx": torch.tensor([b["idx"] for b in batch]),
    }

    if "lang_text_embedding" in batch[0]:
        result["lang_text_embedding"] = torch.stack([b["lang_text_embedding"] for b in batch])
    elif "lang_text" in batch[0]:
        result["lang_text"] = [b["lang_text"][0] for b in batch]

    if "depth_static" in batch[0]:
        result["depth_static"] = torch.stack([b["depth_static"] for b in batch])

    return result


class TianjiDataModule:
    """DataModule adapter for Tianji data, compatible with step2_train_action_calvin.py.

    Provides train_dataloader()["lang"] and val_dataloader()["lang"] interface.
    """

    def __init__(
        self,
        data_dir: str,
        action_col: str = "action",
        action_seq_len: int = 10,
        height: int = 480,
        width: int = 640,
        skip_frames: int = 1,
        batch_size: int = 1,
        num_workers: int = 2,
        val_ratio: float = 0.1,
        seed: int = 42,
        obs_seq_len: int = 1,
        text_embedding_path: str = "",
        depth_data_dir: str = "",
        depth_subpath: str = "exports/mini_npz/depth.mp4",
        **kwargs,
    ):
        self.data_dir = data_dir
        self.action_col = action_col
        self.action_seq_len = action_seq_len
        self.height = height
        self.width = width
        self.skip_frames = skip_frames
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.seed = seed
        self.obs_seq_len = obs_seq_len
        self.text_embedding_path = text_embedding_path
        self.depth_data_dir = depth_data_dir
        self.depth_subpath = depth_subpath
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        full_dataset = TianjiVideoDataset(
            data_dir=self.data_dir,
            action_col=self.action_col,
            action_seq_len=self.action_seq_len,
            height=self.height,
            width=self.width,
            skip_frames=self.skip_frames,
            obs_seq_len=self.obs_seq_len,
            text_embedding_path=self.text_embedding_path,
            depth_data_dir=self.depth_data_dir,
            depth_subpath=self.depth_subpath,
        )
        # Split by episodes: first N% for train, rest for val
        n_episodes = len(full_dataset.episodes)
        n_val = max(1, int(n_episodes * self.val_ratio))
        n_train = n_episodes - n_val

        train_samples = [
            (ep_idx, start)
            for ep_idx, start in full_dataset.sample_index
            if ep_idx < n_train
        ]
        val_samples = [
            (ep_idx, start)
            for ep_idx, start in full_dataset.sample_index
            if ep_idx >= n_train
        ]

        # Reuse the same dataset object, just swap sample_index
        import copy
        self.train_dataset = full_dataset
        self.train_dataset.sample_index = train_samples

        self.val_dataset = copy.copy(full_dataset)
        self.val_dataset.sample_index = val_samples

        print(
            f"TianjiDataModule: {n_train} train episodes ({len(train_samples)} samples), "
            f"{n_val} val episodes ({len(val_samples)} samples)"
        )

    def train_dataloader(self):
        kwargs = {}
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = 2
        return {
            "lang": DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=collate_tianji,
                pin_memory=True,
                **kwargs,
            )
        }

    def val_dataloader(self):
        return {
            "lang": DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collate_tianji,
                pin_memory=True,
            )
        }
