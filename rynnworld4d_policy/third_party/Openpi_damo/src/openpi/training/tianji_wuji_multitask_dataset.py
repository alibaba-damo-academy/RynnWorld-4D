"""Custom dataset for Tianji Wuji multi-task data (parquet + mp4 format, head camera only).

Reads parquet data from multiple task directories and caches it as npy files.
Video frames are read on-the-fly from mp4 files.
"""

import functools
import pathlib
from typing import SupportsIndex

import cv2
import numpy as np
import polars as pl


class TianjiWujiMultiTaskDataset:
    """Dataset for Tianji Wuji multi-task data with head camera only.

    Data format:
        /path/to/data_root/
            Bimanual_Lift/
                episode_000001/
                    observation.images.head.mp4
                    timeseries.parquet
            Pick-Place/
            Push-T/
    """

    DEFAULT_PROMPTS = {
        "Pick-Place": "Use both robotic arms to pick up the banana and apple from the black plate and place them on the white table.",
        "Push-T": "Use both robotic arms to push the blue cube across the white table.",
        "Sort-Can": "Pick up the cabbage with the left gripper, hand it over to the right gripper, and then place it down on the table.",
        "Bimanual_Lift": "Use both robotic arms to grasp the watermelon, lift it from the white table, and reposition it at the center.",
        "Cover-Lid": "Pick up the box lid and place it precisely on top of the cardboard box to cover it.",
        "stack-bowls": "Pick up one bowl and stack it carefully on top of the other bowl on the table.",
        "clean-table": "Clear the table by picking up all three fruits and placing them one by one into the cardboard box.",
    }

    def __init__(
        self,
        data_dir: str,
        *,
        action_horizon: int = 50,
        episodes_per_task: int = 200,
    ):
        self._data_dir = pathlib.Path(data_dir)
        self._action_horizon = action_horizon

        # Detect single-task vs multi-task: if data_dir contains episode_* directly, it's single-task
        direct_episodes = sorted(
            d for d in self._data_dir.iterdir()
            if d.is_dir() and d.name.startswith("episode_")
        )

        if direct_episodes:
            # Single-task mode: data_dir itself is a task directory
            task_dirs = [self._data_dir]
            print(f"Single-task mode: {self._data_dir.name}")
        else:
            # Multi-task mode: subdirectories are tasks
            task_dirs = sorted(
                d for d in self._data_dir.iterdir()
                if d.is_dir() and not d.name.endswith(".zip")
            )
            if not task_dirs:
                raise ValueError(f"No task directories found in {data_dir}")
            print(f"Found {len(task_dirs)} tasks: {[d.name for d in task_dirs]}")

        # Discover episodes per task (limit to episodes_per_task)
        self._episodes = []
        for task_dir in task_dirs:
            episodes = sorted(
                d for d in task_dir.iterdir()
                if d.is_dir() and d.name.startswith("episode_")
            )[:episodes_per_task]
            for ep in episodes:
                self._episodes.append((task_dir.name, ep))

        if not self._episodes:
            raise ValueError(f"No episodes found in {data_dir}")

        print(f"Total episodes: {len(self._episodes)} ({episodes_per_task} per task)")

        # Build cache (always rebuild, no cache loading)
        cache_dir = self._data_dir.parent / ".cache_train"
        cache_dir.mkdir(exist_ok=True)

        states_file = cache_dir / "states.npy"
        actions_file = cache_dir / "actions.npy"
        offsets_file = cache_dir / "offsets.npy"
        prompts_file = cache_dir / "prompts.npy"

        print("Building dataset cache...")
        states_list, actions_list, prompts_list = [], [], []
        offsets = [0]

        for task_name, ep_dir in self._episodes:
            pq = ep_dir / "timeseries.parquet"
            if not pq.exists():
                continue
            df = pl.read_parquet(str(pq))
            states = np.stack(df["observation.state"].to_list()).astype(np.float32)
            actions = np.stack(df["action"].to_list()).astype(np.float32)
            prompt = self.DEFAULT_PROMPTS.get(task_name, "Do the task")

            states_list.append(states)
            actions_list.append(actions)
            prompts_list.append([prompt] * len(states))
            offsets.append(offsets[-1] + len(states))

        self._all_states = np.concatenate(states_list)
        self._all_actions = np.concatenate(actions_list)
        self._prompts = [p for plist in prompts_list for p in plist]
        self._episode_offsets = np.array(offsets, dtype=np.int64)

        np.save(str(states_file), self._all_states)
        np.save(str(actions_file), self._all_actions)
        np.save(str(offsets_file), self._episode_offsets)
        np.save(str(prompts_file), np.array(self._prompts))
        print(f"Cache saved to {cache_dir}")

        self._total_frames = len(self._all_states)
        self._n_episodes = len(self._episode_offsets) - 1

        # Pre-compute video paths
        self._video_paths = [
            str(ep_dir / "observation.images.head.mp4")
            for _, ep_dir in self._episodes
        ]

        print(f"Loaded {self._n_episodes} episodes, {self._total_frames} frames")

    def __len__(self) -> int:
        return self._total_frames

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = index.__index__()
        ep_idx = int(np.searchsorted(self._episode_offsets, idx + 1) - 1)
        local_idx = idx - int(self._episode_offsets[ep_idx])
        ep_len = int(self._episode_offsets[ep_idx + 1]) - int(self._episode_offsets[ep_idx])

        # Action chunk
        end = min(local_idx + self._action_horizon, ep_len)
        action_chunk = self._all_actions[
            int(self._episode_offsets[ep_idx]) + local_idx :
            int(self._episode_offsets[ep_idx]) + local_idx + self._action_horizon
        ]
        if len(action_chunk) < self._action_horizon:
            action_chunk = np.vstack([action_chunk, np.tile(action_chunk[-1:], (self._action_horizon - len(action_chunk), 1))])

        # Video frame
        frame = _read_video_frame(self._video_paths[ep_idx], local_idx)
        if frame is None:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        return {
            "observation/image": frame,
            "observation/state": self._all_states[idx],
            "actions": action_chunk,
            "prompt": self._prompts[idx],
        }


@functools.lru_cache(maxsize=32)
def _get_video_capture(video_path: str):
    if not video_path:
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    return cap


def _read_video_frame(video_path: str, frame_idx: int) -> np.ndarray | None:
    cap = _get_video_capture(video_path)
    if cap is None:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
