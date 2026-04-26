set -euo pipefail

# Moving to repo root regardless of where the script is invoked from.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}/.."

DATA_ROOT="${DATA_ROOT:-/path/to/dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-./path/to/save/weights}"
GPUS="${GPUS:-0,1,3,4}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-29506}"
BACKBONE="${BACKBONE:-facebook/dinov2-base}"

CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
    --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
    train_self_pairing_ddp.py \
    --model "selfcfm" \
    --pixel \
    --lr 2e-4 \
    --pair_lr 1e-4 \
    --ema_decay 0.9999 \
    --batch_size 64 \
    --total_steps 160001 \
    --save_step 40000 \
    --pair_loss_weight 0.10 \
    --pair_warmup_steps 5000 \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --parallel True \
    --master_port "${MASTER_PORT}" \
    --backbone_name "${BACKBONE}" \
    --pair_proj_dim 256 \
    --pair_proj_hidden_dim 768 \
    --pair_temperature 0.07 \
    --pair_conf_threshold 0.20 \
    --pair_lambda_global 1.0 \
    --pair_lambda_patch 1.0 \
    --freeze_backbone True \
    "$@"
