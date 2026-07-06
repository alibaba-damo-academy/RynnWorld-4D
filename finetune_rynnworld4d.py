# Modified from: https://github.com/huggingface/finetrainers
# Training entry point adapted from finetrainers; extended with RynnWorld4D
# multi-branch arguments (joint attention, cosine decay, branch dropout).

from core.finetune.models.utils import get_model_cls
import argparse
import datetime
import logging
from pathlib import Path
from typing import Any, List, Literal, Tuple

from pydantic import BaseModel, ValidationInfo, field_validator


class Args(BaseModel):
    prompt: str = ""

    ########## Model ##########
    model_path: Path
    model_name: str
    model_type: Literal["i2v", "t2v", "i2pm", "i2dpm", "wan-i2v", "egoverse", "egoverse22", "rynnworld4d"]
    training_type: Literal["lora", "sft"] = "lora"

    ########## Output ##########
    output_dir: Path = Path("train_results/{:%Y-%m-%d-%H-%M-%S}".format(datetime.datetime.now()))
    report_to: Literal["tensorboard", "wandb", "all"] | None = None
    tracker_name: str = "finetrainer-cogvideo"

    ########## Data ###########
    # data_root: Path
    # caption_column: Path
    # image_column: Path | None = None
    # video_column: Path

    ########## Training #########
    resume_from_checkpoint: Path | None = None

    seed: int | None = None
    train_epochs: int
    train_steps: int | None = None
    checkpointing_steps: int = 200
    checkpointing_limit: int = 10

    batch_size: int
    gradient_accumulation_steps: int = 1

    train_resolution: Tuple[int, int, int]  # shape: (frames, height, width)

    #### deprecated args: video_resolution_buckets
    # if use bucket for training, should not be None
    # Note1: At least one frame rate in the bucket must be less than or equal to the frame rate of any video in the dataset
    # Note2:  For cogvideox, cogvideox1.5
    #   The frame rate set in the bucket must be an integer multiple of 8 (spatial_compression_rate[4] * path_t[2] = 8)
    #   The height and width set in the bucket must be an integer multiple of 8 (temporal_compression_rate[8])
    # video_resolution_buckets: List[Tuple[int, int, int]] | None = None

    mixed_precision: Literal["no", "fp16", "bf16"]

    learning_rate: float = 2e-5
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.95
    beta3: float = 0.98
    epsilon: float = 1e-8
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    lr_scheduler: str = "cosine_with_warmup"
    lr_warmup_steps: int = 100
    lr_num_cycles: int = 1
    lr_power: float = 1.0

    num_workers: int = 8
    pin_memory: bool = True

    gradient_checkpointing: bool = True
    enable_slicing: bool = True
    enable_tiling: bool = True
    nccl_timeout: int = 1800

    ########## Lora ##########
    rank: int = 128
    lora_alpha: int = 64
    target_modules: List[str] = ["to_q", "to_k", "to_v", "to_out.0"]
    is_concat: bool = True 

    ########## Fusion ##########
    fusion_mode: str = "bidirectional"  # "none", "unidirectional", "bidirectional", "joint"
    share_ffn: bool = True  # True: depth/flow share FFN with RGB; False: independent FFNs
    joint_start_layer: int = 0  # 0-based start layer for joint attention
    joint_end_layer: int = -1  # 0-based exclusive end; -1 means all layers
    joint_every_n_layers: int = 1  # place joint attention every N layers in [start, end)
    joint_frame_wise: bool = False  # restrict cross-modal attention to same-frame tokens
    joint_use_rope: bool = False  # apply 3D RoPE to Q/K in joint cross-modal attention
    joint_unidirectional: bool = False  # if True, video is K/V source only (depth/flow attend to video)
    joint_video_decay: bool = False  # cosine-decay depth/flow→RGB injection to 0 during stage3
    joint_video_decay_steps: int = 700  # steps over which decay goes from 1.0 to 0.0
    joint_out_lr: float = 5e-4  # dedicated LR for zero-init joint_out projection
    joint_other_lr_multiplier: float = 10.0  # multiplier for joint_kv/q/norm/align/modality_embed LR
    resume_from_stage1: str | None = None
    load_stage2_model_weights: str | None = None
    loss_weight_flow: float = 1.0  # Weight for flow branch loss (use <1.0 in stage1 when flow has no informative first frame)

    ########## EMA ##########
    use_ema: bool = False
    ema_decay: float = 0.9999

    ########## Freeze / Branch Dropout ##########
    freeze_non_joint: bool = False
    branch_dropout_prob: float = 0.0
    branch_dropout_modes: List[str] = ["depth", "flow"]

    ########## Validation ##########
    do_validation: bool = False
    validation_steps: int | None  # if set, should be a multiple of checkpointing_steps
    validation_dir: Path | None  # if set do_validation, should not be None
    cache_dir: Path | None
    validation_prompts: str | None  # if set do_validation, should not be None
    validation_images: str | None  # if set do_validation and model_type == i2v, should not be None
    validation_videos: str | None  # if set do_validation and model_type == v2v, should not be None
    gen_fps: int = 15

    ########## Periodic Inference ##########
    periodic_inference_steps: int = 200
    num_inference_samples: int = 3
    inference_num_frames: int = 25
    inference_output_dir: str = "./validation_output"

    #### deprecated args: gen_video_resolution
    # 1. If set do_validation, should not be None
    # 2. Suggest selecting the bucket from `video_resolution_buckets` that is closest to the resolution you have chosen for fine-tuning
    #        or the resolution recommended by the model
    # 3. Note:  For cogvideox, cogvideox1.5
    #        The frame rate set in the bucket must be an integer multiple of 8 (spatial_compression_rate[4] * path_t[2] = 8)
    #        The height and width set in the bucket must be an integer multiple of 8 (temporal_compression_rate[8])
    # gen_video_resolution: Tuple[int, int, int] | None  # shape: (frames, height, width)

    # @field_validator("image_column")
    # def validate_image_column(cls, v: str | None, info: ValidationInfo) -> str | None:
    #     values = info.data
    #     if values.get("model_type") in ["i2v", "i2pm", "i2dpm"] and not v:
    #         logging.warning(
    #             "No `image_column` specified for i2v model. Will automatically extract first frames from videos as conditioning images."
    #         )
    #     return v

    @field_validator("validation_dir", "validation_prompts")
    def validate_validation_required_fields(cls, v: Any, info: ValidationInfo) -> Any:
        values = info.data
        if values.get("do_validation") and not v:
            field_name = info.field_name
            raise ValueError(f"{field_name} must be specified when do_validation is True")
        return v

    @field_validator("validation_images")
    def validate_validation_images(cls, v: str | None, info: ValidationInfo) -> str | None:
        values = info.data
        if values.get("do_validation") and values.get("model_type") in ["i2v", "i2pm", "i2dpm"] and not v:
            raise ValueError("validation_images must be specified when do_validation is True and model_type is i2v")
        return v

    @field_validator("validation_videos")
    def validate_validation_videos(cls, v: str | None, info: ValidationInfo) -> str | None:
        values = info.data
        if values.get("do_validation") and values.get("model_type") == "v2v" and not v:
            raise ValueError("validation_videos must be specified when do_validation is True and model_type is v2v")
        return v

    @field_validator("validation_steps")
    def validate_validation_steps(cls, v: int | None, info: ValidationInfo) -> int | None:
        values = info.data
        if values.get("do_validation"):
            if v is None:
                raise ValueError("validation_steps must be specified when do_validation is True")
            if values.get("checkpointing_steps") and v % values["checkpointing_steps"] != 0:
                raise ValueError("validation_steps must be a multiple of checkpointing_steps")
        return v

    @field_validator("train_resolution")
    def validate_train_resolution(cls, v: Tuple[int, int, int], info: ValidationInfo) -> str:
        try:
            frames, height, width = v

            # Check if (frames - 1) is multiple of 8
            if (frames - 1) % 8 != 0:
                raise ValueError("Number of frames - 1 must be a multiple of 8")

            # Check resolution for cogvideox-5b models
            model_name = info.data.get("model_name", "")
            if model_name in ["cogvideox-5b-i2v", "cogvideox-5b-t2v"]:
                if (height, width) != (480, 720):
                    raise ValueError("For cogvideox-5b models, height must be 480 and width must be 720")

            return v

        except ValueError as e:
            if (
                str(e) == "not enough values to unpack (expected 3, got 0)"
                or str(e) == "invalid literal for int() with base 10"
            ):
                raise ValueError("train_resolution must be in format 'frames x height x width'")
            raise e

    @field_validator("mixed_precision")
    def validate_mixed_precision(cls, v: str, info: ValidationInfo) -> str:
        if v == "fp16" and "cogvideox-2b" not in str(info.data.get("model_path", "")).lower():
            logging.warning(
                "All CogVideoX models except cogvideox-2b were trained with bfloat16. "
                "Using fp16 precision may lead to training instability."
            )
        return v

    @classmethod
    def parse_args(cls):
        """Parse command line arguments and return Args instance"""
        parser = argparse.ArgumentParser()
        # Required arguments
        parser.add_argument("--prompt", type=str, default="", required=False)
        parser.add_argument("--model_path", type=str, required=True)
        parser.add_argument("--model_name", type=str, required=True)
        parser.add_argument("--model_type", type=str, required=True)
        parser.add_argument("--training_type", type=str, required=True)
        parser.add_argument("--output_dir", type=str, required=True)
        # parser.add_argument("--data_root", type=str, required=True)
        # parser.add_argument("--caption_column", type=str, required=True)
        # parser.add_argument("--video_column", type=str, required=True)
        parser.add_argument("--train_resolution", type=str, required=True)
        parser.add_argument("--report_to", type=str, required=True)

        # Training hyperparameters
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--train_epochs", type=int, default=10)
        parser.add_argument("--train_steps", type=int, default=None)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--optimizer", type=str, default="adamw")
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.95)
        parser.add_argument("--beta3", type=float, default=0.98)
        parser.add_argument("--epsilon", type=float, default=1e-8)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--max_grad_norm", type=float, default=1.0)

        # Learning rate scheduler
        parser.add_argument("--lr_scheduler", type=str, default="cosine_with_warmup")
        parser.add_argument("--lr_warmup_steps", type=int, default=500)
        parser.add_argument("--lr_num_cycles", type=int, default=1)
        parser.add_argument("--lr_power", type=float, default=1.0)

        # Data loading
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--pin_memory", type=bool, default=True)
        parser.add_argument("--image_column", type=str, default=None)

        # Model configuration
        parser.add_argument("--mixed_precision", type=str, default="no")
        parser.add_argument("--gradient_checkpointing", type=lambda x: (str(x).lower() == 'true'), default=True)
        parser.add_argument("--enable_slicing", type=lambda x: (str(x).lower() == 'true'), default=True)
        parser.add_argument("--enable_tiling", type=lambda x: (str(x).lower() == 'true'), default=True)
        parser.add_argument("--nccl_timeout", type=int, default=1800)

        # LoRA parameters
        parser.add_argument("--rank", type=int, default=128)
        parser.add_argument("--lora_alpha", type=int, default=64)
        parser.add_argument("--target_modules", type=str, nargs="+", default=["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0","attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0","ffn.net.0.proj", "ffn.net.2","proj_out"])

        # Checkpointing
        parser.add_argument("--checkpointing_steps", type=int, default=200)
        parser.add_argument("--checkpointing_limit", type=int, default=10)
        parser.add_argument("--resume_from_checkpoint", type=str, default=None)

        # Validation
        parser.add_argument("--do_validation", type=lambda x: x.lower() == 'true', default=False)
        parser.add_argument("--validation_steps", type=int, default=None)
        parser.add_argument("--validation_dir", type=str, default=None)
        parser.add_argument("--cache_dir", type=str, default=None)
        parser.add_argument("--validation_prompts", type=str, default=None)
        parser.add_argument("--validation_images", type=str, default=None)
        parser.add_argument("--validation_videos", type=str, default=None)
        parser.add_argument("--gen_fps", type=int, default=15)

        # Periodic inference during training
        parser.add_argument("--periodic_inference_steps", type=int, default=200)
        parser.add_argument("--num_inference_samples", type=int, default=3)
        parser.add_argument("--inference_num_frames", type=int, default=25)
        parser.add_argument("--inference_output_dir", type=str, default="./validation_output")


        parser.add_argument("--is_concat", type=lambda x: (str(x).lower() == 'true'), default=True)
        parser.add_argument("--fusion_mode", type=str, default="bidirectional",
                            choices=["none", "unidirectional", "bidirectional", "joint"],
                            help="Fusion mode: none=no cross-branch fusion, unidirectional=depth/flow→video only, bidirectional=all directions, joint=joint self-attention across branches")
        parser.add_argument("--share_ffn", type=lambda x: (str(x).lower() == 'true'), default=True,
                            help="Whether depth/flow branches share FFN with RGB branch. Set to False for independent FFNs.")
        parser.add_argument("--joint_start_layer", type=int, default=0,
                            help="0-based start layer for joint attention. Only used when fusion_mode=joint.")
        parser.add_argument("--joint_end_layer", type=int, default=-1,
                            help="0-based exclusive end layer for joint attention; -1 means all layers. Only used when fusion_mode=joint.")
        parser.add_argument("--joint_every_n_layers", type=int, default=1,
                            help="Place joint attention every N layers inside [joint_start_layer, joint_end_layer). Only used when fusion_mode=joint.")
        parser.add_argument("--joint_frame_wise", type=lambda x: (str(x).lower() == 'true'), default=False,
                            help="If True, restrict cross-modal joint attention to tokens from the same frame. Prevents motion blur from cross-time mixing.")
        parser.add_argument("--joint_use_rope", type=lambda x: (str(x).lower() == 'true'), default=False,
                            help="If True, apply 3D RoPE to Q/K in joint cross-modal attention so position is respected across modalities.")
        parser.add_argument("--joint_unidirectional", type=lambda x: (str(x).lower() == 'true'), default=False,
                            help="If True, video acts as the K/V source only and is NOT modified by joint attention; depth/flow each attend solely to video's K/V. Prevents weaker depth/flow features from contaminating the strong RGB branch.")
        parser.add_argument("--joint_video_decay", type=lambda x: (str(x).lower() == 'true'), default=False,
                            help="If True, cosine-decay the depth/flow→RGB injection (modality_embed_video + gate) from 1.0 to 0.0 over joint_video_decay_steps. Only meaningful with joint_unidirectional=False (bidirectional).")
        parser.add_argument("--joint_video_decay_steps", type=int, default=700,
                            help="Number of optimizer steps over which the video decay goes from 1.0 to 0.0.")
        parser.add_argument("--joint_out_lr", type=float, default=5e-4,
                            help="Dedicated learning rate for the zero-initialized joint_out projection. Only used when fusion_mode=joint.")
        parser.add_argument("--joint_other_lr_multiplier", type=float, default=10.0,
                            help="Multiplier applied to learning_rate for joint_kv/q/norm/align/modality_embed params. Only used when fusion_mode=joint.")
        parser.add_argument("--resume_from_stage1", type=str, default=None,
                            help="Path to stage1 checkpoint directory to load pretrained depth/flow weights before stage2 training")
        parser.add_argument("--load_stage2_model_weights", type=str, default=None,
                            help="Path to a stage2 checkpoint directory (saved with different GPU count) to load model weights only, skipping ZeRO optimizer states. Useful when world size differs.")
        parser.add_argument("--loss_weight_flow", type=float, default=1.0,
                            help="Weight for flow branch loss. Use <1.0 in stage1 when flow has no informative first frame.")
        parser.add_argument("--use_ema", type=lambda x: (str(x).lower() == 'true'), default=False,
                            help="Whether to use Exponential Moving Average of model weights.")
        parser.add_argument("--ema_decay", type=float, default=0.9999,
                            help="EMA decay rate.")
        parser.add_argument("--freeze_non_joint", type=lambda x: (str(x).lower() == 'true'), default=False,
                            help="Freeze all parameters except joint attention (joint_out/kv/q/norm/gate). Forces optimizer to only update cross-modal layers.")
        parser.add_argument("--branch_dropout_prob", type=float, default=0.0,
                            help="Probability (0-1) to randomly corrupt one branch's noisy latents (except the first frame) during training. Forces joint attention to use cross-modal information.")
        parser.add_argument("--branch_dropout_modes", type=str, default="depth,flow",
                            help="Comma-separated list of branches that can be dropped out: video,depth,flow. Usually keep video and only drop depth/flow.")

        args = parser.parse_args()
        args.branch_dropout_modes = [m.strip() for m in args.branch_dropout_modes.split(",") if m.strip()]

        # Convert video_resolution_buckets string to list of tuples
        frames, height, width = args.train_resolution.split("x")
        args.train_resolution = (int(frames), int(height), int(width))

        return cls(**vars(args))


def main():
    args = Args.parse_args()
    trainer_cls = get_model_cls(args.model_name, args.training_type)
    trainer = trainer_cls(args)
    trainer.fit()


if __name__ == "__main__":
    main()