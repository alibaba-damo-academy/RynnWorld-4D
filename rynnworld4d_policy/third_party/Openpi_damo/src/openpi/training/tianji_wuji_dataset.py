"""Custom dataset for Tianji Wuji Pick-Place data (parquet + mp4 format).

Reads parquet data and caches it as npy files for fast subsequent loads.
Video frames are read on-the-fly from mp4 files.
"""

import functools
import pathlib
from typing import SupportsIndex

import cv2
import numpy as np
import polars as pl


class TianjiWujiDataset:
    """Dataset for Tianji Wuji Pick-Place data.

    Data format:
        /path/to/Pick-Place/
            episode_000001/
                observation.images.head.mp4
                timeseries.parquet
            ...

    Caches state/action as .npy on first load for fast subsequent runs.
    """

    # Default prompt used when no prompt is available in the data.
    DEFAULT_PROMPT = "pick up the object and place it"

    def __init__(
        self,
        data_dir: str,
        *,
        action_horizon: int = 50,
    ):
        self._data_dir = pathlib.Path(data_dir)
        self._action_horizon = action_horizon

        # Discover all episodes
        self._episodes = sorted(
            d for d in self._data_dir.iterdir()
            if d.is_dir() and d.name.startswith("episode_")
        )

        if not self._episodes:
            raise ValueError(f"No episodes found in {data_dir}")

        # Try to load cached data, or build from parquet
        cache_dir = self._data_dir / ".cache"
        cache_dir.mkdir(exist_ok=True)

        states_cache = cache_dir / "states.npy"
        actions_cache = cache_dir / "actions.npy"
        offsets_cache = cache_dir / "offsets.npy"

        if states_cache.exists() and actions_cache.exists() and offsets_cache.exists():
            self._all_states = np.load(str(states_cache))
            self._all_actions = np.load(str(actions_cache))
            self._episode_offsets = np.load(str(offsets_cache))
        else:
            print("Building dataset cache from parquet files...")
            states_list = []
            actions_list = []
            offsets = [0]

            for ep_dir in self._episodes:
                parquet_path = ep_dir / "timeseries.parquet"
                if not parquet_path.exists():
                    continue

                df = pl.read_parquet(str(parquet_path))
                states = np.stack(df["observation.state"].to_list()).astype(np.float32)
                actions = np.stack(df["action"].to_list()).astype(np.float32)

                states_list.append(states)
                actions_list.append(actions)
                offsets.append(offsets[-1] + len(states))

            self._all_states = np.concatenate(states_list, axis=0)
            self._all_actions = np.concatenate(actions_list, axis=0)
            self._episode_offsets = np.array(offsets, dtype=np.int64)

            np.save(str(states_cache), self._all_states)
            np.save(str(actions_cache), self._all_actions)
            np.save(str(offsets_cache), self._episode_offsets)
            print(f"Cache saved to {cache_dir}")

        # Map frame index to episode
        self._total_frames = len(self._all_states)
        self._n_episodes = len(self._episode_offsets) - 1

        # Pre-compute video paths per episode
        self._video_paths = []
        for ep_dir in self._episodes:
            video_path = ep_dir / "observation.images.head.mp4"
            if video_path.exists():
                self._video_paths.append(str(video_path))
            else:
                self._video_paths.append("")

        print(f"Loaded {self._n_episodes} episodes, {self._total_frames} frames")

    def __len__(self) -> int:
        return self._total_frames

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = index.__index__()

        # Find episode
        ep_idx = int(np.searchsorted(self._episode_offsets, idx + 1, side='right') - 1)
        local_idx = idx - int(self._episode_offsets[ep_idx])

        # Get action chunk (future frames)
        start = local_idx
        end = min(local_idx + self._action_horizon, int(self._episode_offsets[ep_idx + 1]) - int(self._episode_offsets[ep_idx]))
        action_chunk = self._all_actions[int(self._episode_offsets[ep_idx]) + start : int(self._episode_offsets[ep_idx]) + start + self._action_horizon]

        # Pad if needed
        if len(action_chunk) < self._action_horizon:
            pad = np.tile(action_chunk[-1:], (self._action_horizon - len(action_chunk), 1))
            action_chunk = np.vstack([action_chunk, pad])

        # Read video frame
        video_path = self._video_paths[ep_idx]
        image = _read_video_frame(video_path, local_idx) if video_path else None
        if image is None:
            image = np.zeros((720, 1280, 3), dtype=np.uint8)

        return {
            "observation/image": image,
            "observation/state": self._all_states[idx],
            "actions": action_chunk,
            "prompt": self.DEFAULT_PROMPT,
        }


@functools.lru_cache(maxsize=16)
def _get_video_capture(video_path: str):
    """Cached video capture to avoid reopening files."""
    if not video_path:
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    return cap


def _read_video_frame(video_path: str, frame_idx: int) -> np.ndarray | None:
    """Read a single frame from a video file."""
    cap = _get_video_capture(video_path)
    if cap is None:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    # Convert BGR to RGB
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
