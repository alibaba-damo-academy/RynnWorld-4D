#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=eth0  # Set to your network interface
export NCCL_CROSS_NIC=1
export NCCL_IB_TIMEOUT=22

set -ex

# --- Derive PROJECT_ROOT from script location so the repo is relocatable ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "START TIME: $(date)"
echo "Running on host: $(hostname)"

if [ -z "$WORLD_SIZE" ] || [ -z "$RANK" ] || [ -z "$MASTER_ADDR" ] || [ -z "$MASTER_PORT" ] || [ -z "$NPROC_PER_NODE" ]; then
    echo "CRITICAL ERROR: Platform did not set required env vars for torchrun."
    exit 1
fi

# --- Cross-node MASTER_PORT sync via rendezvous file on NAS ---
RENDEZVOUS_FILE="${PROJECT_ROOT}/.rendezvous_${MASTER_ADDR}.port"
if [ "$RANK" = "0" ]; then
    rm -f "$RENDEZVOUS_FILE"
    mkdir -p "$(dirname "$RENDEZVOUS_FILE")"
    echo "$MASTER_PORT" > "$RENDEZVOUS_FILE"
    echo "[stage1] rank0 MASTER_PORT=$MASTER_PORT, wrote to $RENDEZVOUS_FILE"
else
    echo "[stage1] rank$RANK waiting for $RENDEZVOUS_FILE ..."
    for i in $(seq 1 600); do
        if [ -f "$RENDEZVOUS_FILE" ]; then
            RANK0_PORT=$(cat "$RENDEZVOUS_FILE" | tr -d '[:space:]')
            if [ -n "$RANK0_PORT" ]; then
                export MASTER_PORT="$RANK0_PORT"
                echo "[stage1] rank$RANK got rank0 MASTER_PORT=$RANK0_PORT"
                break
            fi
        fi
        sleep 2
    done
    if [ -z "$RANK0_PORT" ]; then
        echo "ERROR: rank$RANK timed out waiting for rendezvous file $RENDEZVOUS_FILE"
        exit 1
    fi
fi

# --- Install dependencies ---
# Install ffmpeg if not available
which ffmpeg >/dev/null 2>&1 || { apt-get update && apt-get install -y ffmpeg || yum install -y ffmpeg; } || true

PIP_MIRROR=""  # Set to your pip mirror if needed (e.g. "-i https://pypi.tuna.tsinghua.edu.cn/simple/")
pip install -r ${PROJECT_ROOT}/requirements.txt $PIP_MIRROR

pip install -e . --no-build-isolation $PIP_MIRROR

# --- Training ---
TOTAL_GPUS=$((WORLD_SIZE * NPROC_PER_NODE))

echo "--- Correcting Accelerate Launch Arguments ---"
echo "Total Processes (TOTAL_GPUS): $TOTAL_GPUS"
echo "Total Machines (NNODES): $WORLD_SIZE"
echo "Current Machine Rank (NODE_RANK): $RANK"
echo "MASTER: $MASTER_ADDR : $MASTER_PORT"
echo "---------------------------------------------"

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false
export ACCELERATE_CONFIG_FILE="configs_acc/multinode_deepspeed.yaml"

PROGRAM_FILE="finetune_rynnworld4d.py"

MODEL_ARGS=(
    --model_path ${PROJECT_ROOT}/pretrained/Wan2.2-TI2V-5B-Diffusers
    --model_name rynnworld4d
    --model_type rynnworld4d
)

TRAINING_ARGS=(
    --training_type sft
    --train_epochs 1
    --seed 42
    --batch_size 1
    --gradient_accumulation_steps 4
    --mixed_precision bf16
    --num_workers 8
    --pin_memory True
    --nccl_timeout 7200
    --gradient_checkpointing True
)

OPTIMIZER_ARGS=(
    --learning_rate 1.5e-5
    --lr_scheduler cosine_with_warmup
    --lr_warmup_steps 300
)

# --- Resolve "latest" checkpoint to actual path ---
OUTPUT_DIR="${PROJECT_ROOT}/training/rynnworld4d-stage1-${TOTAL_GPUS}gpu-v2"
RESUME_CHECKPOINT="latest"

if [ "$RESUME_CHECKPOINT" = "latest" ]; then
    LATEST_CKPT=""
    if [ -d "$OUTPUT_DIR" ]; then
        for d in "$OUTPUT_DIR"/checkpoint-*; do
            [ -d "$d" ] || continue
            STEP_NUM=$(basename "$d" | sed 's/checkpoint-//')
            case "$STEP_NUM" in
                ''|*[!0-9]*) continue ;;
            esac
            if [ -z "$LATEST_CKPT" ] || [ "$STEP_NUM" -gt "${LATEST_CKPT##*-}" ]; then
                LATEST_CKPT="$d"
            fi
        done
    fi
    if [ -n "$LATEST_CKPT" ]; then
        RESUME_CHECKPOINT="$LATEST_CKPT"
        echo "[stage1] Resolved 'latest' to: $RESUME_CHECKPOINT"
    else
        RESUME_CHECKPOINT=""
        echo "[stage1] No checkpoints found in $OUTPUT_DIR, starting from scratch"
    fi
fi

LOG_ARGS=(
    --output_dir "$OUTPUT_DIR"
    --report_to tensorboard
    --checkpointing_steps 200
    --checkpointing_limit 50
)

if [ -n "$RESUME_CHECKPOINT" ]; then
    LOG_ARGS+=(--resume_from_checkpoint "$RESUME_CHECKPOINT")
fi

DATA_ARGS=(
    --train_resolution 25x480x832
    --do_validation false
    --validation_dir ${PROJECT_ROOT}/data/sample.json
    --cache_dir ${PROJECT_ROOT}/data/sample_latents
    --prompt ''
    --is_concat True
)

FLOWWORLD_ARGS=(
    --fusion_mode none
    --share_ffn False
    --use_ema True
    --loss_weight_flow 0.5
    --periodic_inference_steps 200
    --num_inference_samples 3
)

accelerate launch \
    --config_file "$ACCELERATE_CONFIG_FILE" \
    --num_processes "$TOTAL_GPUS" \
    --num_machines "$WORLD_SIZE" \
    --machine_rank "$RANK" \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    $PROGRAM_FILE \
    "${MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${LOG_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${FLOWWORLD_ARGS[@]}"

echo "END TIME: $(date)"
