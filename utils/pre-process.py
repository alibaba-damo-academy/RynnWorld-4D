from diffusers import (
    AutoencoderKLWan,
)
from transformers import T5TokenizerFast, UMT5EncoderModel
import torch
import json
from pathlib import Path
from safetensors.torch import save_file
import decord
from torchvision import transforms
from termcolor import cprint
import os
import math
import regex as re
import ftfy
import html
import subprocess
import tempfile


def transcode_to_h264(video_path: str, tmp_dir: str = None) -> str:
    if tmp_dir is None:
        tmp_dir = os.path.join(tempfile.gettempdir(), "rynnworld4d_transcode")
    os.makedirs(tmp_dir, exist_ok=True)
    out_path = os.path.join(tmp_dir, f"{os.getpid()}_{os.path.basename(video_path)}")
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast', '-crf', '18', '-an',
        out_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


class PreProcess:
    def __init__(self, width, height, max_num_frames, model_path='./pretrained/Wan2.2-TI2V-5B-Diffusers'):
        self.model_path = model_path
        self.device = torch.device("cuda")
        self.width = width
        self.height = height
        self.max_num_frames = max_num_frames
        self.__transforms = transforms.Compose([
            transforms.CenterCrop((self.height, self.width)),
            transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)
        ])
        self.load_components()

    def load_components(self):
        self.tokenizer = T5TokenizerFast.from_pretrained(self.model_path, subfolder="tokenizer")
        self.text_encoder = UMT5EncoderModel.from_pretrained(self.model_path, subfolder="text_encoder").to(self.device)
        self.vae = AutoencoderKLWan.from_pretrained(self.model_path, subfolder="vae").to(self.device)
        cprint('load all components to device: {}'.format(self.device), 'green')

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        vae = self.vae
        video = video.to(vae.device, dtype=vae.dtype)
        with torch.no_grad():
            latent_dist = vae.encode(video).latent_dist
            latent = latent_dist.mode()
            latents_mean = (
                torch.tensor(vae.config.latents_mean)
                .view(1, vae.config.z_dim, 1, 1, 1)
                .to(latent.device, latent.dtype)
            )
            latents_std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
                latent.device, latent.dtype
            )
            latent = (latent - latents_mean) / latents_std
        return latent

    def _get_t5_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        def basic_clean(text):
            text = ftfy.fix_text(text)
            text = html.unescape(html.unescape(text))
            return text.strip()

        def whitespace_clean(text):
            text = re.sub(r"\s+", " ", text)
            text = text.strip()
            return text

        def prompt_clean(text):
            text = whitespace_clean(basic_clean(text))
            return text

        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: str | list[str],
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_embedding = self._get_t5_prompt_embeds(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

        return prompt_embedding
    
    def preprocess(self, data_root, cache_dir, slice, id, keyword, args):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        data_root = Path(data_root)
        with open(data_root, 'r', encoding='utf-8') as f:
            full_data = json.load(f)

        total_items = len(full_data)
        items_per_slice = math.ceil(total_items / slice)
        start_index = id * items_per_slice
        end_index = min(start_index + items_per_slice, total_items)
        current_shard_data = full_data[start_index:end_index]
        num = len(current_shard_data)

        cprint(f"Shard {id}: Processing {num} videos (Total: {total_items})", "cyan")

        filename_with_ext = os.path.basename(data_root)
        basename, _ = os.path.splitext(filename_with_ext)
        manifest_path = cache_dir / f"{basename}_{slice}_{id}.json"

        data_manifest = []
        processed_count = 0
        skipped_count = 0

        for index, item in enumerate(current_shard_data):
            if index % 10 == 0:
                cprint(f"Index: {index}/{num} (processed: {processed_count}, skipped: {skipped_count}, manifest: {len(data_manifest)})", "cyan")

            if index % 50 == 0 and index > 0:
                with open(manifest_path, 'w', encoding='utf-8') as f:
                    json.dump(data_manifest, f, indent=2)

            video_path_str = item.get('video_path') or item.get('path')

            if keyword not in video_path_str:
                skipped_count += 1
                continue

            sub_path = video_path_str.split(keyword)[-1].lstrip('/')
            video_path_obj = Path(sub_path)
            output_base_dir = Path(cache_dir) / video_path_obj.parent / video_path_obj.stem
            # Robust mkdir for NFS: retry on transient errors
            for _attempt in range(3):
                try:
                    os.makedirs(str(output_base_dir), exist_ok=True)
                    break
                except (FileExistsError, FileNotFoundError, OSError):
                    import time
                    time.sleep(0.1)
            else:
                # Final fallback: if it's already a dir, that's fine
                if not output_base_dir.is_dir():
                    raise

            caps = item.get('caption', [])
            if not caps:
                skipped_count += 1
                continue

            if isinstance(caps[0], dict):
                unified_caps = [
                    (c['start_time'], c['end_time'], c['description'])
                    for c in caps
                    if c.get('start_time') is not None
                    and c.get('end_time') is not None
                    and c.get('description')
                ]
            else:
                unified_caps = [(0.0, None, str(c)) for c in caps if c]

            if not unified_caps:
                skipped_count += 1
                continue

            all_done = False
            temp_entries = []
            if unified_caps[0][1] is not None:
                all_done = True
                for s_t, e_t, prompt in unified_caps:
                    existing_rgb = sorted(
                        p for p in output_base_dir.glob(f"{s_t}_{e_t}_*.safetensors")
                        if '_flow_depth' not in p.name
                    )
                    if len(existing_rgb) == 0:
                        all_done = False
                        break
                    if args.encode_flow_depth:
                        for rgb_file in existing_rgb:
                            fd_file = rgb_file.parent / rgb_file.name.replace('.safetensors', '_flow_depth.safetensors')
                            if not (fd_file.exists() and fd_file.stat().st_size > 0):
                                all_done = False
                                break
                        if not all_done:
                            break
                    for rgb_file in existing_rgb:
                        entry = {
                            "rgb_latents": str(rgb_file),
                            "video_path": video_path_str,
                            "prompt": prompt,
                        }
                        if args.encode_flow_depth:
                            fd_file = rgb_file.parent / rgb_file.name.replace('.safetensors', '_flow_depth.safetensors')
                            entry["flow_depth_latents"] = str(fd_file)
                        temp_entries.append(entry)

            if all_done and temp_entries:
                data_manifest.extend(temp_entries)
                processed_count += 1
                continue

            video_path = Path(video_path_str)
            if not video_path.exists():
                cprint(f"Warning: Video not found: {video_path_str}", "red")
                skipped_count += 1
                continue

            effective_video_path = str(video_path)
            transcoded_path = None
            try:
                tmp_vr = decord.VideoReader(uri=effective_video_path)
                fps = tmp_vr.get_avg_fps()
                total_frames = len(tmp_vr)
                orig_h, orig_w = tmp_vr[0].shape[:2]
                del tmp_vr
            except Exception as e:
                try:
                    cprint(f"  -> Transcoding (decord failed): {video_path_str}", "yellow")
                    transcoded_path = transcode_to_h264(effective_video_path)
                    effective_video_path = transcoded_path
                    tmp_vr = decord.VideoReader(uri=effective_video_path)
                    fps = tmp_vr.get_avg_fps()
                    total_frames = len(tmp_vr)
                    orig_h, orig_w = tmp_vr[0].shape[:2]
                    del tmp_vr
                except Exception as e2:
                    cprint(f"Metadata error: {video_path_str}, {e2}", "red")
                    if transcoded_path and os.path.exists(transcoded_path):
                        os.remove(transcoded_path)
                    skipped_count += 1
                    continue

            scale = max(self.width / orig_w, self.height / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)

            vr = vr_depth = vr_flow = None
            try:
                vr = decord.VideoReader(uri=effective_video_path, width=new_w, height=new_h)

                if args.encode_flow_depth:
                    rel_path = video_path_str.split(keyword)[-1]
                    sub_dir = os.path.splitext(rel_path)[0]
                    depth_video_path = os.path.join(args.depth_root, sub_dir, "exports/mini_npz", "depth.mp4")
                    flow_video_path = os.path.join(args.flow_root, sub_dir, "flow.mp4")

                    missing = False
                    for p, name in [(depth_video_path, "depth"), (flow_video_path, "flow")]:
                        if not os.path.exists(p):
                            cprint(f"Skip: {name} not found: {p}", "yellow")
                            missing = True
                            break
                    if missing:
                        continue

                    vr_depth = decord.VideoReader(uri=str(depth_video_path), width=new_w, height=new_h)
                    vr_flow = decord.VideoReader(uri=str(flow_video_path), width=new_w, height=new_h)

                video_duration = total_frames / fps
                unified_caps = [
                    (s_t, video_duration if e_t is None else e_t, prompt)
                    for s_t, e_t, prompt in unified_caps
                ]

                if unified_caps[0][1] == video_duration and len(temp_entries) == 0:
                    num_chunks_check = max(1, total_frames // self.max_num_frames)
                    remainder_check = total_frames % self.max_num_frames
                    if remainder_check > 40:
                        num_chunks_check += 1
                    s_t_ck, e_t_ck = unified_caps[0][0], unified_caps[0][1]
                    all_chunks_exist = True
                    for c_idx in range(num_chunks_check):
                        chk = output_base_dir / f"{s_t_ck}_{e_t_ck}_{c_idx}.safetensors"
                        if not (chk.exists() and chk.stat().st_size > 0):
                            all_chunks_exist = False
                            break
                        if args.encode_flow_depth:
                            fd_chk = output_base_dir / f"{s_t_ck}_{e_t_ck}_{c_idx}_flow_depth.safetensors"
                            if not (fd_chk.exists() and fd_chk.stat().st_size > 0):
                                all_chunks_exist = False
                                break
                    if all_chunks_exist:
                        for s_t_ck, e_t_ck, prompt in unified_caps:
                            for c_idx in range(num_chunks_check):
                                entry = {
                                    "rgb_latents": str(output_base_dir / f"{s_t_ck}_{e_t_ck}_{c_idx}.safetensors"),
                                    "video_path": video_path_str,
                                    "prompt": prompt,
                                }
                                if args.encode_flow_depth:
                                    entry["flow_depth_latents"] = str(output_base_dir / f"{s_t_ck}_{e_t_ck}_{c_idx}_flow_depth.safetensors")
                                data_manifest.append(entry)
                        cprint(f"All chunks exist for: {video_path_obj.stem}", "green")
                        processed_count += 1
                        continue

                text_cache = {}

                for start_t, end_t, prompt in unified_caps:

                    if prompt not in text_cache:
                        text_cache[prompt] = self.encode_prompt(prompt, device=self.device).cpu()
                    text_embeds = text_cache[prompt]

                    start_frame = int(start_t * fps)
                    end_frame = min(int(end_t * fps), total_frames)
                    duration_frames = end_frame - start_frame

                    if duration_frames < self.max_num_frames:
                        continue

                    num_chunks = duration_frames // self.max_num_frames
                    remainder = duration_frames % self.max_num_frames
                    if remainder > 40:
                        num_chunks += 1

                    for chunk_idx in range(num_chunks):
                        c_start = start_frame + chunk_idx * self.max_num_frames
                        c_end = c_start + self.max_num_frames

                        if c_end > end_frame:
                            c_end = end_frame
                            c_start = max(start_frame, c_end - self.max_num_frames)

                        rgb_path = output_base_dir / f"{start_t}_{end_t}_{chunk_idx}.safetensors"
                        fd_path = (output_base_dir / f"{start_t}_{end_t}_{chunk_idx}_flow_depth.safetensors") if args.encode_flow_depth else None

                        rgb_done = rgb_path.exists() and rgb_path.stat().st_size > 0
                        fd_done = (not args.encode_flow_depth) or (fd_path.exists() and fd_path.stat().st_size > 0)
                        if rgb_done and fd_done:
                            entry = {"rgb_latents": str(rgb_path), "video_path": video_path_str, "prompt": prompt}
                            if args.encode_flow_depth:
                                entry["flow_depth_latents"] = str(fd_path)
                            data_manifest.append(entry)
                            continue

                        try:
                            frame_indices = list(range(c_start, c_end))
                            video_frames = torch.from_numpy(vr.get_batch(frame_indices).asnumpy()).permute(0, 3, 1, 2)

                            depth_frames = None
                            flow_frames = None
                            if args.encode_flow_depth:
                                depth_frames = torch.from_numpy(vr_depth.get_batch(frame_indices).asnumpy()).permute(0, 3, 1, 2)
                                flow_indices = list(range(c_start, c_end - 1))
                                flow_frames = torch.from_numpy(vr_flow.get_batch(flow_indices).asnumpy()).permute(0, 3, 1, 2)
                                zero_flow_frame = torch.full_like(flow_frames[:1], 255)
                                flow_frames = torch.cat([zero_flow_frame, flow_frames], dim=0)
                        except Exception as e:
                            cprint(f"Error fetching frames {c_start}-{c_end}: {e}", "red")
                            continue

                        self._process_and_save(video_frames, text_embeds, rgb_path,
                                               depth_frames, flow_frames, fd_path)

                        entry = {"rgb_latents": str(rgb_path), "video_path": video_path_str, "prompt": prompt}
                        if args.encode_flow_depth:
                            entry["flow_depth_latents"] = str(fd_path)
                        data_manifest.append(entry)
                        torch.cuda.empty_cache()

            except Exception as e:
                cprint(f"Process error: {video_path_str}, {e}", "red")
                import traceback
                traceback.print_exc()
                skipped_count += 1
            else:
                processed_count += 1
            finally:
                for obj in [vr, vr_depth, vr_flow]:
                    if obj is not None:
                        del obj
                if transcoded_path and os.path.exists(transcoded_path):
                    os.remove(transcoded_path)

        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(data_manifest, f, indent=2)
        cprint(f"\nDone! Processed {processed_count}/{num} videos, skipped {skipped_count}, manifest entries: {len(data_manifest)}", "green")
        cprint(f"Manifest saved to: {manifest_path}", "green")

    def _process_and_save(self, video_frames, text_embeds, rgb_path,
                          depth_frames=None, flow_frames=None, fd_path=None):
        rgb_path.parent.mkdir(parents=True, exist_ok=True)

        # --- Encode RGB ---
        if not (rgb_path.exists() and rgb_path.stat().st_size > 0):
            frames = self.video_transform(video_frames.float())
            video_input = frames.permute(1, 0, 2, 3).unsqueeze(0).to(self.device, dtype=self.vae.dtype)
            with torch.no_grad():
                latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(self.device)
                latents_std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1).to(self.device)
                video_latents = self.vae.encode(video_input).latent_dist.mode()
                video_latents = (video_latents - latents_mean) / latents_std
            print('video_latents',video_latents.shape)
            print('text_embeds',text_embeds.shape)
            data_to_save = {
                "video_latents": video_latents.squeeze(0).cpu().contiguous(),
                "text_embeds": text_embeds.contiguous(),
            }
            for _ in range(3):
                save_file(data_to_save, str(rgb_path))
                if rgb_path.exists() and rgb_path.stat().st_size > 0:
                    break

        # --- Encode depth / flow ---
        if fd_path is not None and depth_frames is not None and flow_frames is not None:
            if not (fd_path.exists() and fd_path.stat().st_size > 0):
                d_frames = self.video_transform(depth_frames.float())
                f_frames = self.video_transform(flow_frames.float())
                depth_input = d_frames.permute(1, 0, 2, 3).unsqueeze(0).to(self.device, dtype=self.vae.dtype)
                flow_input = f_frames.permute(1, 0, 2, 3).unsqueeze(0).to(self.device, dtype=self.vae.dtype)
                with torch.no_grad():
                    latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(self.device)
                    latents_std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1).to(self.device)
                    depth_latents = self.vae.encode(depth_input).latent_dist.mode()
                    depth_latents = (depth_latents - latents_mean) / latents_std
                    flow_latents = self.vae.encode(flow_input).latent_dist.mode()
                    flow_latents = (flow_latents - latents_mean) / latents_std
                print('flow_latents',flow_latents.shape)
                print('depth_latents',depth_latents.shape)
                fd_data = {
                    "flow_latents": flow_latents.squeeze(0).cpu().contiguous(),
                    "depth_latents": depth_latents.squeeze(0).cpu().contiguous(),
                }
                for _ in range(3):
                    save_file(fd_data, str(fd_path))
                    if fd_path.exists() and fd_path.stat().st_size > 0:
                        break

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__transforms(f) for f in frames], dim=0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="preprocess data for training")
    parser.add_argument("--data_root", type=str, default='./data/sample.json')
    parser.add_argument("--cache_dir", type=str, default='./data/cache/')
    parser.add_argument("--keyword", type=str, default="RDT-1B/")
    parser.add_argument("--slice", type=int, default=10)
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--max_num_frames", type=int, default=81)
    parser.add_argument("--encode_flow_depth", action="store_true", help="additionally encode depth and optical flow videos")
    parser.add_argument("--depth_root", type=str, default="", help="root dir for depth videos (required if --encode_flow_depth)")
    parser.add_argument("--flow_root", type=str, default="", help="root dir for flow videos (required if --encode_flow_depth)")

    args = parser.parse_args()

    processer = PreProcess(
        width=640,
        height=480,
        max_num_frames=args.max_num_frames,
    )
    processer.preprocess(
        data_root=args.data_root, cache_dir=args.cache_dir,
        slice=args.slice, id=args.id, keyword=args.keyword, args=args
    )
