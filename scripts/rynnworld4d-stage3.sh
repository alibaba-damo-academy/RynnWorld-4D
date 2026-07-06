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
RENDEZVOUS_FILE="${PROJECT_ROOT}/.rendezvous_stage3_${MASTER_ADDR}.port"
START_TS=$(date +%s)
if [ "$RANK" = "0" ]; then
    rm -f "$RENDEZVOUS_FILE"
    mkdir -p "$(dirname "$RENDEZVOUS_FILE")"
    TMP_FILE="${RENDEZVOUS_FILE}.tmp.$$"
    echo "$MASTER_PORT" > "$TMP_FILE"
    mv -f "$TMP_FILE" "$RENDEZVOUS_FILE"
    echo "[stage3] rank0 MASTER_PORT=$MASTER_PORT, wrote to $RENDEZVOUS_FILE (start_ts=$START_TS)"
else
    echo "[stage3] rank$RANK waiting for fresh $RENDEZVOUS_FILE (start_ts=$START_TS) ..."
    MIN_MTIME=$((START_TS - 300))
    for i in $(seq 1 600); do
        if [ -f "$RENDEZVOUS_FILE" ]; then
            FILE_MTIME=$(stat -c %Y "$RENDEZVOUS_FILE" 2>/dev/null || echo 0)
            if [ "$FILE_MTIME" -ge "$MIN_MTIME" ]; then
                RANK0_PORT=$(cat "$RENDEZVOUS_FILE" | tr -d '[:space:]')
                if [ -n "$RANK0_PORT" ]; then
                    export MASTER_PORT="$RANK0_PORT"
                    echo "[stage3] rank$RANK got rank0 MASTER_PORT=$RANK0_PORT (file_mtime=$FILE_MTIME)"
                    break
                fi
            else
                if [ $((i % 15)) -eq 0 ]; then
                    echo "[stage3] rank$RANK ignoring stale rendezvous file (mtime=$FILE_MTIME < $MIN_MTIME)"
                fi
            fi
        fi
        sleep 2
    done
    if [ -z "$RANK0_PORT" ]; then
        echo "ERROR: rank$RANK timed out waiting for fresh rendezvous file $RENDEZVOUS_FILE"
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

echo "--- Stage 3: Full-parameter Fine-tuning ---"
echo "Total Processes (TOTAL_GPUS): $TOTAL_GPUS"
echo "Total Machines (NNODES): $WORLD_SIZE"
echo "Current Machine Rank (NODE_RANK): $RANK"
echo "MASTER: $MASTER_ADDR : $MASTER_PORT"
echo "---------------------------------------------"

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Activate DeepSpeed inside Accelerator() via env vars (torchrun launch, not accelerate launch)
export ACCELERATE_USE_DEEPSPEED=true
export ACCELERATE_DEEPSPEED_CONFIG_FILE="configs_zero/zero2_offload.yaml"
export ACCELERATE_DEEPSPEED_ZERO3_INIT=false
export ACCELERATE_MIXED_PRECISION=bf16

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
    --gradient_accumulation_steps 2
    --mixed_precision bf16
    --num_workers 8
    --pin_memory True
    --nccl_timeout 7200
    --gradient_checkpointing True
)

OPTIMIZER_ARGS=(
    --learning_rate 5e-6
    --joint_out_lr 1e-5
    --joint_other_lr_multiplier 1.0
    --lr_scheduler cosine_with_warmup
    --lr_warmup_steps 500
)

# --- Resolve checkpoint strategy ---
OUTPUT_DIR="${PROJECT_ROOT}/training/rynnworld4d-stage3-${TOTAL_GPUS}gpu-unidirectional"

# Helper: find latest checkpoint in a directory
find_latest_checkpoint() {
    local dir="$1"
    local latest=""
    if [ -d "$dir" ]; then
        for d in "$dir"/checkpoint-*; do
            [ -d "$d" ] || continue
            local step
            step=$(basename "$d" | sed 's/checkpoint-//')
            case "$step" in
                ''|*[!0-9]*) continue ;;
            esac
            if [ -z "$latest" ] || [ "$step" -gt "${latest##*-}" ]; then
                latest="$d"
            fi
        done
    fi
    echo "$latest"
}

# If we already have stage3 checkpoints, resume (with optimizer state).
# Otherwise start fresh from stage2 checkpoint (model weights only, fresh optimizer).
RESUME_CHECKPOINT=$(find_latest_checkpoint "$OUTPUT_DIR")

STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-${PROJECT_ROOT}/training/rynnworld4d-stage2/checkpoint-latest}"

if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "[stage3] Resuming from existing stage3 checkpoint (with optimizer): $RESUME_CHECKPOINT"
    LOG_ARGS=(--resume_from_checkpoint "$RESUME_CHECKPOINT")
    LOAD_STAGE2_ARGS=()
else
    echo "[stage3] Starting fresh from stage2: $STAGE2_CHECKPOINT (model weights only, fresh optimizer)"
    LOG_ARGS=()
    LOAD_STAGE2_ARGS=(
        --load_stage2_model_weights "$STAGE2_CHECKPOINT"
    )
fi

LOG_ARGS+=(
    --output_dir "$OUTPUT_DIR"
    --report_to tensorboard
    --checkpointing_steps 100
    --checkpointing_limit 40
)

DATA_ARGS=(
    --train_resolution 25x480x832
    --do_validation false
    --validation_dir ${PROJECT_ROOT}/data/sample.json
    --cache_dir ${PROJECT_ROOT}/data/sample_latents
    --prompt ''
    --is_concat True
)

FLOWWORLD_ARGS=(
    --fusion_mode joint
    --share_ffn False
    --joint_start_layer 0
    --joint_end_layer 30
    --joint_every_n_layers 3
    --joint_frame_wise True
    --joint_use_rope True
    --joint_unidirectional True
    --use_ema True
    --ema_decay 0.999
    --loss_weight_flow 1.0
    --periodic_inference_steps 200
    --num_inference_samples 3
    --freeze_non_joint False
    --branch_dropout_prob 0.05
    --branch_dropout_modes depth,flow
)

echo "============================================"
echo "[stage3] Full-parameter + Joint Attention"
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "  Resume (with optimizer): $RESUME_CHECKPOINT"
else
    echo "  Start from Stage2: $STAGE2_CHECKPOINT"
fi
echo "  Output dir:  $OUTPUT_DIR"
echo "  Training: ALL parameters (transformer + joint + branches)"
echo "  backbone lr=5e-6, joint_other_mult=1.0 (=> 5e-6), joint_out_lr=1e-5"
echo "  warmup=500, eff_batch=1*2*GPUs=$((TOTAL_GPUS*2))"
echo "  RoPE in joint attention: ENABLED"
echo "  EMA: ENABLED"
echo "  Data: rynnworld4d-all-with-tianji-add.json (3.17M samples)"
echo "============================================"

DEBUG_LOG_DIR="${PROJECT_ROOT}/training/rynnworld4d-stage3-${TOTAL_GPUS}gpu-unidirectional/debug_logs"
mkdir -p "$DEBUG_LOG_DIR"
DEBUG_LOG_FILE="$DEBUG_LOG_DIR/rank${RANK}_$(date +%Y%m%d-%H%M%S).log"
echo "[debug] writing detailed log to $DEBUG_LOG_FILE"

{
echo "[debug] which python: $(which python)"
echo "[debug] which accelerate: $(which accelerate)"
echo "[debug] which torchrun: $(which torchrun)"
python - <<'PYEOF'
import sys
print('[debug] python', sys.version, flush=True)
try:
    import torch
    print('[debug] torch', torch.__version__, 'cuda', torch.version.cuda, flush=True)
    print('[debug] torch.cuda.device_count', torch.cuda.device_count(), flush=True)
except Exception as e:
    print('[debug] torch import failed:', repr(e), flush=True)
try:
    import accelerate
    print('[debug] accelerate', accelerate.__version__, flush=True)
except Exception as e:
    print('[debug] accelerate import failed:', repr(e), flush=True)
try:
    import deepspeed
    print('[debug] deepspeed', deepspeed.__version__, flush=True)
except Exception as e:
    print('[debug] deepspeed import failed:', repr(e), flush=True)
PYEOF
PY_EXIT=$?
echo "[debug] python sanity exit code: $PY_EXIT"
echo "[debug] RANK=$RANK WORLD_SIZE=$WORLD_SIZE NPROC_PER_NODE=$NPROC_PER_NODE MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "[debug] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[debug] nvidia-smi -L:"
nvidia-smi -L || echo "[debug] nvidia-smi failed"
} 2>&1 | tee -a "$DEBUG_LOG_FILE"

echo "[debug] starting torchrun, output -> $DEBUG_LOG_FILE"
set +e
torchrun \
    --nnodes "$WORLD_SIZE" \
    --node_rank "$RANK" \
    --nproc_per_node "$NPROC_PER_NODE" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    --rdzv-conf timeout=3600 \
    --redirects 3 \
    --tee 3 \
    --log_dir "$DEBUG_LOG_DIR/torchrun_rank${RANK}" \
    $PROGRAM_FILE \
    "${MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${LOG_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${FLOWWORLD_ARGS[@]}" \
    "${LOAD_STAGE2_ARGS[@]}" 2>&1 | tee -a "$DEBUG_LOG_FILE"
TORCHRUN_EXIT_CODE=${PIPESTATUS[0]}
set -e
echo "[debug] torchrun exit code: $TORCHRUN_EXIT_CODE" | tee -a "$DEBUG_LOG_FILE"
echo "[debug] last 50 lines of $DEBUG_LOG_FILE:"
tail -n 50 "$DEBUG_LOG_FILE" || true
echo "[debug] per-rank torchrun logs in $DEBUG_LOG_DIR/torchrun_rank${RANK}:"
find "$DEBUG_LOG_DIR/torchrun_rank${RANK}" -type f 2>/dev/null | while read -r f; do
    echo "===== $f ====="
    tail -n 100 "$f" || true
done

echo "END TIME: $(date)"

if [ "$TORCHRUN_EXIT_CODE" -ne 0 ]; then
    exit "$TORCHRUN_EXIT_CODE"
fi