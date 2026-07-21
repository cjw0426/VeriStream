#!/bin/bash
# OVO-Bench evaluation runner for Qwen3-VL / adapter variants.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
EVAL_SCRIPT="${EVAL_SCRIPT:-main_experiments/eval_qwen3vl_ovo.py}"
DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"
QWEN_EXACT_RECENT_DECODE="${QWEN_EXACT_RECENT_DECODE:-1}"
FRAME_SELECTION="${FRAME_SELECTION:-recent}" # recent / uniform / clip_topk / recent_uniform / recent_clip_topk / recent_memory_uniform / recent_memory_clip_topk / recent_state_memory_uniform_v4 / recent_state_memory_clip_topk_v4 / recent_state_memory_stratified_v4 / recent_vst_memory_uniform / recent_vst_memory_clip_topk / recent_vst_memory_uniform_v2 / recent_vst_memory_clip_topk_v2
SUPPLEMENTAL_FRAMES="${SUPPLEMENTAL_FRAMES:-0}"
MEMORY_NUM_ITEMS="${MEMORY_NUM_ITEMS:-3}"
MEMORY_GROUP_SIZE="${MEMORY_GROUP_SIZE:-4}"
MEMORY_CLIP_SIZE="${MEMORY_CLIP_SIZE:-4}"
MEMORY_MAX_CLIPS="${MEMORY_MAX_CLIPS:-4}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-openai/clip-vit-large-patch14}"
CLIP_DEVICE="${CLIP_DEVICE:-auto}"
CLIP_BATCH_SIZE="${CLIP_BATCH_SIZE:-32}"

cd "${REPO_ROOT}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES="$(
        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t',' -k2,2nr \
        | head -n "${NUM_PROCESSES}" \
        | cut -d',' -f1 \
        | tr -d ' ' \
        | paste -sd, -
    )"
fi

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Using EVAL_SCRIPT=${EVAL_SCRIPT}"
echo "Using DECORD_EOF_RETRY_MAX=${DECORD_EOF_RETRY_MAX}"
echo "Using QWEN_EXACT_RECENT_DECODE=${QWEN_EXACT_RECENT_DECODE}"
echo "Using FRAME_SELECTION=${FRAME_SELECTION}"
echo "Using SUPPLEMENTAL_FRAMES=${SUPPLEMENTAL_FRAMES}"
echo "Using MEMORY_NUM_ITEMS=${MEMORY_NUM_ITEMS}"
echo "Using MEMORY_GROUP_SIZE=${MEMORY_GROUP_SIZE}"
echo "Using MEMORY_CLIP_SIZE=${MEMORY_CLIP_SIZE}"
echo "Using MEMORY_MAX_CLIPS=${MEMORY_MAX_CLIPS}"

CMD=(
    "${PYTHON_BIN}" -m accelerate.commands.launch
    --num_processes "${NUM_PROCESSES}"
    --multi_gpu
    --mixed_precision bf16
)

CMD+=(--main_process_port "${MAIN_PROCESS_PORT:-0}")

CMD+=(
    "${EVAL_SCRIPT}"
    --model_path "${MODEL_PATH:-/data1/chenjunwei/models/Qwen3-VL-8B-Instruct}"
    --anno_path "${OVO_ANNO_PATH:-/data1/chenjunwei/projects/SimpleStream/data/ovo_bench/ovo_bench_new.json}"
    --chunked_dir "${OVO_CHUNKED_DIR:-/data1/chenjunwei/projects/SimpleStream/data/ovo_bench/chunked_videos}"
    --result_dir "${OVO_RESULT_DIR:-main_experiments/results/ovo_qwen3vl_recent4}"
    --frame_selection "${FRAME_SELECTION}"
    --recent_frames_only "${RECENT_FRAMES_ONLY:-4}"
    --supplemental_frames "${SUPPLEMENTAL_FRAMES}"
    --memory_num_items "${MEMORY_NUM_ITEMS}"
    --memory_group_size "${MEMORY_GROUP_SIZE}"
    --memory_clip_size "${MEMORY_CLIP_SIZE}"
    --memory_max_clips "${MEMORY_MAX_CLIPS}"
    --chunk_duration "${CHUNK_DURATION:-1.0}"
    --fps "${FPS:-1.0}"
    --clip_model_path "${CLIP_MODEL_PATH}"
    --clip_device "${CLIP_DEVICE}"
    --clip_batch_size "${CLIP_BATCH_SIZE}"
    --max_qa_tokens "${MAX_QA_TOKENS:-256}"
)

if [[ -n "${MEMORY_MAX_TOKENS:-}" ]]; then
    CMD+=(--memory_max_tokens "${MEMORY_MAX_TOKENS}")
fi

if [[ -n "${ADAPTER_PATH:-}" ]]; then
    CMD+=(--adapter_path "${ADAPTER_PATH}")
fi

if [[ -n "${MAX_SAMPLES_PER_SPLIT:-}" ]]; then
    CMD+=(--max_samples_per_split "${MAX_SAMPLES_PER_SPLIT}")
fi

if [[ -n "${EVAL_SPLITS:-}" ]]; then
    CMD+=(--eval_splits "${EVAL_SPLITS}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX}" \
QWEN_EXACT_RECENT_DECODE="${QWEN_EXACT_RECENT_DECODE}" \
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" \
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}" \
"${CMD[@]}"
