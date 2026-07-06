#!/usr/bin/env python3
"""
Randomly sample 50 frames from flow videos of each dataset and save as images.
Datasets: EgoVid, RoboMIND, Galaxea
Output: <repo>/vis/flow/{dataset}/<random_frame>.jpg

Set env vars to point at your per-dataset flow-video manifest JSONs:
  EGOVID_FLOW_JSON, ROBOMIND_FLOW_JSON, GALAXEA_FLOW_JSON
"""

import json
import os
import random
import subprocess
import cv2 as cv
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_BASE = Path(os.environ.get("FLOW_VIS_OUTPUT", REPO_ROOT / "vis" / "flow"))
NUM_SAMPLES = 50


def get_frame_count(vid_path):
    """Get frame count using ffprobe (much faster than cv2 on NFS)."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_packets", "-show_entries", "stream=nb_read_packets",
             "-of", "csv=p=0", vid_path],
            capture_output=True, text=True, timeout=10
        )
        n = int(result.stdout.strip())
        if n > 0:
            return n
    except:
        pass
    # Fallback to cv2
    cap = cv.VideoCapture(vid_path)
    if not cap.isOpened():
        cap.release()
        return 0
    n = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def extract_frame(vid_path, frame_idx, out_path):
    """Extract a single frame from video and save as jpg."""
    # Use ffprobe to seek to the frame, faster than cv2 on NFS
    # Try cv2 as it's more reliable for random access
    cap = cv.VideoCapture(vid_path)
    if not cap.isOpened():
        cap.release()
        return False
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv.imwrite(str(out_path), frame, [cv.IMWRITE_JPEG_QUALITY, 90])
        return True
    return False


def sample_dataset(videos, dataset_name, n_videos=50):
    """Sample NUM_SAMPLES frames from a list of videos."""
    out_dir = OUTPUT_BASE / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  {dataset_name}: {len(videos)} videos total, selecting {n_videos}")
    selected = random.sample(videos, min(n_videos, len(videos)))
    
    # Get frame counts for selected videos
    video_frames = []  # (vid_path, n_frames)
    for vid_path in selected:
        n = get_frame_count(vid_path)
        if n > 0:
            video_frames.append((vid_path, n))
    
    if not video_frames:
        print(f"  No valid videos found!")
        return 0
    
    # Collect all possible (video, frame_idx) pairs, then randomly pick NUM_SAMPLES
    all_pairs = []
    for vid_path, n_frames in video_frames:
        # Pick 1-2 random frames per video to spread across videos
        n_pick = min(max(1, NUM_SAMPLES // len(video_frames) + 1), n_frames)
        indices = random.sample(range(n_frames), n_pick)
        for idx in indices:
            all_pairs.append((vid_path, idx))
    
    # Randomly pick exactly NUM_SAMPLES
    if len(all_pairs) > NUM_SAMPLES:
        all_pairs = random.sample(all_pairs, NUM_SAMPLES)
    
    print(f"  Extracting {len(all_pairs)} frames...")
    
    saved = 0
    for i, (vid_path, frame_idx) in enumerate(all_pairs):
        vid_name = Path(vid_path).stem.replace(" ", "_")[:40]
        out_path = out_dir / f"{dataset_name}_{saved:03d}_{vid_name}_f{frame_idx:06d}.jpg"
        if extract_frame(vid_path, frame_idx, out_path):
            saved += 1
            print(f"  [{saved}/{len(all_pairs)}] Saved: {out_path.name}")
    
    print(f"  Saved {saved} images to {out_dir}")
    return saved


def load_and_sample_json(json_path, n_sample, key=None):
    """Load a JSON file and randomly sample n_sample video paths."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if key:
        data = data.get(key, [])
    
    total = len(data)
    n = min(n_sample, total)
    
    # For EgoVid: list of dicts with 'path' key
    if isinstance(data[0], dict):
        all_paths = [item.get('path') or item.get('video', {}).get('path') for item in data]
    else:
        all_paths = data
    
    # Random sample first, then filter to existing files (avoids iterating millions)
    candidates = random.sample(all_paths, n)
    
    existing = []
    for p in candidates:
        try:
            if os.path.exists(p):
                existing.append(p)
        except:
            pass
    
    print(f"  {total} total, sampled {n}, {len(existing)} exist on disk", flush=True)
    return existing


def main():
    random.seed(42)
    t0 = time.time()

    egovid_json = os.environ.get("EGOVID_FLOW_JSON")
    robomind_json = os.environ.get("ROBOMIND_FLOW_JSON")
    galaxea_json = os.environ.get("GALAXEA_FLOW_JSON")

    # === EgoVid ===
    if egovid_json:
        print("\n=== Sampling EgoVid ===", flush=True)
        egovid_videos = load_and_sample_json(egovid_json, n_sample=200)
        sample_dataset(egovid_videos, "egovid", n_videos=50)
    else:
        print("[skip] EgoVid: set EGOVID_FLOW_JSON to its flow manifest", flush=True)

    # === RoboMIND ===
    if robomind_json:
        print("\n=== Sampling RoboMIND ===", flush=True)
        robomind_videos = load_and_sample_json(robomind_json, n_sample=200)
        sample_dataset(robomind_videos, "robomind", n_videos=50)
    else:
        print("[skip] RoboMIND: set ROBOMIND_FLOW_JSON to its flow manifest", flush=True)
    
    # === Galaxea ===
    if galaxea_json:
        print("\n=== Sampling Galaxea ===", flush=True)
        galaxea_videos = load_and_sample_json(galaxea_json, n_sample=200, key="head_rgb")
        sample_dataset(galaxea_videos, "galaxea", n_videos=50)
    else:
        print("[skip] Galaxea: set GALAXEA_FLOW_JSON to its flow manifest", flush=True)
    
    print(f"\n=== Done in {time.time()-t0:.1f}s ===", flush=True)
    for d in ["egovid", "robomind", "galaxea"]:
        dpath = OUTPUT_BASE / d
        n = len(list(dpath.glob("*.jpg"))) if dpath.is_dir() else 0
        print(f"  {d}: {n} images in {dpath}")


if __name__ == "__main__":
    main()
