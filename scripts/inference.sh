set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}/.."

: "${CKPT_PATH:?Set CKPT_PATH=/path/to/checkpoint.pt}"
DATA_ROOT="${DATA_ROOT:-/path/to/dataset}"
OUT_DIR="${OUT_DIR:-./fid_runs}"
GPUS="${GPUS:-0}"
PYTHON="${PYTHON:-python3.11}"
BACKBONE="${BACKBONE:-facebook/dinov2-base}"

CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON}" compute_fid.py \
    --model "selfcfm" \
    --pixel \
    --ckpt_path "${CKPT_PATH}" \
    --data_root "${DATA_ROOT}" \
    --out_dir "${OUT_DIR}" \
    --batch_size 32 \
    --num_gen 5000 \
    --integration_method dopri5 \
    --integration_steps 50 \
    --backbone_name "${BACKBONE}" \
    "$@"
