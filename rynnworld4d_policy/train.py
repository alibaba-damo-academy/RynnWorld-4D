"""
Stage-2 training for Tianji robot data with Wan backbone.

Usage:
    # Single GPU
    python step2_train_tianji.py

    # Multi-GPU via accelerate
    accelerate launch step2_train_tianji.py

    # Override data path
    python step2_train_tianji.py --root_data_dir /path/to/data
"""
import gc
import logging
import os
import sys
import traceback
from pathlib import Path
from time import time

import torch
from accelerate import Accelerator
from hydra import compose, initialize
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))

import hydra as hydra_lib


def print_mem(label=""):
    """Print GPU and CPU memory usage."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[MEM] {label} GPU: alloc={alloc:.2f}GB, reserved={reserved:.2f}GB")
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # GB on Linux
        print(f"[MEM] {label} CPU RSS peak: {rss:.2f}GB")
    except:
        pass


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{logging_dir}/log.txt"),
        ],
    )
    return logging.getLogger(__name__)


@torch.no_grad()
def update_ema_trainable(ema_params_dict, model, decay=0.9999):
    """EMA update for trainable parameters only (skip frozen backbone)."""
    for name, param in model.named_parameters():
        if param.requires_grad and name in ema_params_dict:
            ema_params_dict[name].mul_(decay).add_(param.data, alpha=1 - decay)


def train(cfg):
    os.environ['HYDRA_FULL_ERROR'] = '1'
    print_mem("START")

    accelerator = Accelerator()
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    device = accelerator.device
    torch.set_float32_matmul_precision('medium')
    print_mem("After Accelerator init")

    if accelerator.is_main_process:
        os.makedirs(cfg.log_dir, exist_ok=True)
        from datetime import datetime
        uuid = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        experiment_dir = f"{cfg.log_dir}/{uuid}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")

    # --- Model (create BEFORE datamodule to avoid DataLoader worker memory during SFT loading)
    if accelerator.is_main_process:
        logger.info("[DEBUG] Instantiating model...")
        print_mem("Before model instantiate")
    t0 = time()
    model = hydra_lib.utils.instantiate(cfg.model)
    if accelerator.is_main_process:
        logger.info(f"[DEBUG] Model instantiated in {time()-t0:.1f}s")
        print_mem("After model instantiate")

    if cfg.use_ckpt_path:
        if accelerator.is_main_process:
            logger.info(f"[DEBUG] Loading checkpoint: {cfg.ckpt_path}")
            print_mem("Before ckpt load")
        t0 = time()
        state_dict = torch.load(cfg.ckpt_path, map_location='cpu')
        print(f'load_from_ckpt: {cfg.ckpt_path}')
        model.load_state_dict(state_dict['model'])
        del state_dict
        gc.collect()
        if accelerator.is_main_process:
            logger.info(f"[DEBUG] Checkpoint loaded in {time()-t0:.1f}s")
            print_mem("After ckpt load")

    if accelerator.is_main_process:
        logger.info("[DEBUG] Moving model to device...")
        print_mem("Before model.to(device)")
    t0 = time()
    model = model.to(device)
    if accelerator.is_main_process:
        logger.info(f"[DEBUG] Model moved to device in {time()-t0:.1f}s")
        print_mem("After model.to(device)")

    model.process_device()
    if accelerator.is_main_process:
        print_mem("After process_device()")

    if accelerator.is_main_process:
        n_total = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total params: {n_total/1e6:.1f}M, Trainable: {n_train/1e6:.1f}M")
        logger.info(f"GPU mem after build: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # --- DataModule
    if accelerator.is_main_process:
        logger.info("[DEBUG] Instantiating datamodule...")
        print_mem("Before datamodule")
    t0 = time()
    datamodule = hydra_lib.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    if accelerator.is_main_process:
        logger.info(f"[DEBUG] Datamodule setup in {time()-t0:.1f}s")
        print_mem("After datamodule setup")

    t0 = time()
    train_loader = datamodule.train_dataloader()["lang"]
    if accelerator.is_main_process:
        logger.info(f"[DEBUG] Train dataloader created in {time()-t0:.1f}s")
        print_mem("After train_loader creation")
    test_loader = datamodule.val_dataloader()["lang"]

    if accelerator.is_main_process:
        logger.info(f"Train loader: {len(train_loader)} batches, Val loader: {len(test_loader)} batches")

    # --- Optimizer
    if accelerator.is_main_process:
        logger.info("[DEBUG] Configuring optimizers...")
        print_mem("Before configure_optimizers")
    t0 = time()
    opt_dict = model.configure_optimizers()
    if accelerator.is_main_process:
        logger.info(f"[DEBUG] Optimizers configured in {time()-t0:.1f}s")
        print_mem("After configure_optimizers")
    opt = opt_dict["optimizer"]
    lr_scheduler = opt_dict["lr_scheduler"]["scheduler"]

    # --- EMA (only trainable params, not the frozen 11.5B backbone)
    if accelerator.is_main_process:
        logger.info("[DEBUG] Setting up EMA...")
        print_mem("Before EMA")
    t0 = time()
    model.on_train_start()
    ema_params = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    ema_bytes = sum(p.numel() * p.element_size() for p in ema_params.values())
    if accelerator.is_main_process:
        logger.info(f"[DEBUG] EMA setup in {time()-t0:.1f}s, tracking {len(ema_params)} params ({ema_bytes/1e6:.1f}MB)")
        print_mem("After EMA")

    # --- Accelerate prepare
    if accelerator.is_main_process:
        logger.info("[DEBUG] Preparing with accelerate...")
        print_mem("Before accelerator.prepare")
    t0 = time()
    model.train()
    model, opt, loader = accelerator.prepare(model, opt, train_loader)
    if accelerator.is_main_process:
        logger.info(f"[DEBUG] Accelerator prepare in {time()-t0:.1f}s")
        print_mem("After accelerator.prepare")

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    best_eval_loss = 1e8

    if accelerator.is_main_process:
        logger.info(f"Training for {cfg.max_epochs} epochs...")

    for epoch in range(cfg.max_epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        running_loss = 0

        limit = getattr(cfg, "limit_train_batches", None)
        for idx, data_batch in enumerate(loader):
            if idx == 0 and accelerator.is_main_process:
                print_mem("Before first batch")
            if limit and idx >= limit:
                break

            with accelerator.autocast():
                loss = model(data_batch)

            if idx == 0 and accelerator.is_main_process:
                print_mem("After forward pass")

            opt.zero_grad()
            accelerator.backward(loss)

            if idx == 0 and accelerator.is_main_process:
                print_mem("After backward")

            # Gradient clipping
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )

            opt.step()
            lr_scheduler.step()
            update_ema_trainable(ema_params, model)

            if idx == 0 and accelerator.is_main_process:
                print_mem("After optimizer step")

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % cfg.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = running_loss / log_steps

                if accelerator.is_main_process:
                    logger.info(
                        f"(step={train_steps:07d}) Loss: {avg_loss:.6f}, "
                        f"Steps/Sec: {steps_per_sec:.2f}, "
                        f"GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB"
                    )
                running_loss = 0
                log_steps = 0
                start_time = time()

        # --- Validation
        if accelerator.is_main_process:
            model.eval()
            logger.info(f"Validation epoch {epoch}...")
            total_val_loss = 0
            val_steps = 0
            val_limit = getattr(cfg, "limit_val_batches", 50)
            for vi, test_batch in enumerate(test_loader):
                if vi >= val_limit:
                    break
                val_loss = model.module.validation_step(test_batch) if accelerator.num_processes > 1 else model.validation_step(test_batch)
                total_val_loss += val_loss["validation_loss"]
                val_steps += 1
            model.train()

            total_val_loss = total_val_loss / max(val_steps, 1)
            logger.info(f"Epoch {epoch} val_loss: {total_val_loss:.6f}")

            # --- Checkpoint
            checkpoint = {
                "model": model.module.state_dict() if accelerator.num_processes > 1 else model.state_dict(),
                "ema": ema_params,
                "args": cfg,
                "epoch": epoch,
                "train_steps": train_steps,
            }
            if total_val_loss < best_eval_loss:
                best_path = f"{checkpoint_dir}/{train_steps:07d}_{total_val_loss:.3f}.pt"
                torch.save(checkpoint, best_path)
                logger.info(f"New best! Saved to {best_path}")
                best_eval_loss = total_val_loss

            last_path = f"{checkpoint_dir}/last.pt"
            torch.save(checkpoint, last_path)
            logger.info(f"Saved last checkpoint to {last_path}")


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_data_dir", type=str, default="")
    parser.add_argument("--wan_model_path", type=str, default="")
    parser.add_argument("--config", type=str, default="train_config",
                        help="Config name in policy_conf/")
    args = parser.parse_args()

    with initialize(config_path="./policy_conf", job_name="tianji_train"):
        cfg = compose(config_name=args.config)

    if args.root_data_dir:
        cfg.root_data_dir = args.root_data_dir
    if args.wan_model_path:
        cfg.model.pretrained_model_path = args.wan_model_path

    try:
        train(cfg)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        print_mem("AT CRASH")
        print(f"{'='*60}\n")
        sys.exit(1)
