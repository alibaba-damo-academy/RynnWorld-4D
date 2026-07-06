#!/bin/bash
# Train RynnWorld4D-Policy (Stage-2 robot policy on top of frozen RynnWorld4D backbone).

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=eth0  # Set to your network interface
export NCCL_CROSS_NIC=1
export NCCL_IB_TIMEOUT=22
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID

set -ex

# --- Derive PROJECT_ROOT from script location so the repo is relocatable ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "START TIME: $(date)"
echo "Running on host: $(hostname)"

POLICY_DIR="${PROJECT_ROOT}/rynnworld4d_policy"

if [ -n "$WORLD_SIZE" ] && [ -n "$RANK" ] && [ -n "$MASTER_ADDR" ] && [ -n "$MASTER_PORT" ] && [ -n "$NPROC_PER_NODE" ]; then
    # --- Cross-node MASTER_PORT sync via rendezvous file on NAS ---
    RENDEZVOUS_FILE="${PROJECT_ROOT}/.rendezvous_policy_${MASTER_ADDR}.port"
    if [ "$RANK" = "0" ]; then
        rm -f "$RENDEZVOUS_FILE"
        mkdir -p "$(dirname "$RENDEZVOUS_FILE")"
        echo "$MASTER_PORT" > "$RENDEZVOUS_FILE"
        echo "[policy-train] rank0 MASTER_PORT=$MASTER_PORT, wrote to $RENDEZVOUS_FILE"
    else
        echo "[policy-train] rank$RANK waiting for $RENDEZVOUS_FILE ..."
        for i in $(seq 1 600); do
            if [ -f "$RENDEZVOUS_FILE" ]; then
                RANK0_PORT=$(cat "$RENDEZVOUS_FILE" | tr -d '[:space:]')
                if [ -n "$RANK0_PORT" ]; then
                    export MASTER_PORT="$RANK0_PORT"
                    echo "[policy-train] rank$RANK got rank0 MASTER_PORT=$RANK0_PORT"
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

    TOTAL_GPUS=$((WORLD_SIZE * NPROC_PER_NODE))
    DISTRIBUTED=1
else
    echo "[policy-train] No multi-node env vars detected, running single-node."
    DISTRIBUTED=0
    TOTAL_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
fi

# --- Install dependencies (optional, comment out if already installed) ---
if [ "${SKIP_PIP_INSTALL:-0}" != "1" ]; then
    PIP_MIRROR=""  # Set to your pip mirror if needed
    # Install ffmpeg if not available
    which ffmpeg >/dev/null 2>&1 || { apt-get update && apt-get install -y ffmpeg || yum install -y ffmpeg; } || true
    pip install -r "${POLICY_DIR}/requirements.txt" $PIP_MIRROR || true
fi

# --- Run training (train.py uses Hydra and loads ./policy_conf relative to CWD) ---
cd "${POLICY_DIR}"

# Forward extra args to train.py (e.g. --root_data_dir ..., --wan_model_path ..., --config ...)
TRAIN_ARGS=("$@")

if [ "$DISTRIBUTED" = "1" ]; then
    echo "--- Multi-node launch ---"
    echo "Total Processes (TOTAL_GPUS): $TOTAL_GPUS"
    echo "Total Machines (NNODES): $WORLD_SIZE"
    echo "Current Machine Rank (NODE_RANK): $RANK"
    echo "MASTER: $MASTER_ADDR : $MASTER_PORT"

    accelerate launch \
        --num_processes "$TOTAL_GPUS" \
        --num_machines "$WORLD_SIZE" \
        --machine_rank "$RANK" \
        --main_process_ip "$MASTER_ADDR" \
        --main_process_port "$MASTER_PORT" \
        --mixed_precision bf16 \
        train.py "${TRAIN_ARGS[@]}"
else
    echo "--- Single-node launch ($TOTAL_GPUS GPU(s)) ---"
    if [ "$TOTAL_GPUS" -gt 1 ]; then
        accelerate launch --num_processes "$TOTAL_GPUS" --mixed_precision bf16 train.py "${TRAIN_ARGS[@]}"
    else
        python train.py "${TRAIN_ARGS[@]}"
    fi
fi

echo "END TIME: $(date)"
