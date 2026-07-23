# SimpleStream

SimpleStream is a training-free long-video understanding agent built on frozen Qwen3-VL models. The
main method, **VeriStream**, organizes a causal video prefix as persistent evidence memory and lets a
bounded LLM controller decide whether to search, inspect, expand, compare, or stop. The repository
evaluates OVO-Bench **Backward** and **Realtime** tasks; Forward is outside the primary experiment.

The publication figure should be generated with the reviewed
[GPT Image 2 method prompt](docs/veristream/gpt_image2_method_figure_prompt.md) and saved as
`figures/veristream_method.png` after visual inspection. The prompt includes a correction pass and a
logical-consistency checklist; no generated placeholder figure is committed.

## Method At A Glance

```text
offline per-video indexing
  L0 raw frames + timestamps + CLIP embeddings
    -> temporal coverage + multi-span change proposals
    -> frozen-VLM semantic assessment
    -> L1 OverviewMemory / admitted TransitionMemory with L0 provenance

question time
  question -> atomic EvidenceNeeds -> LLM tool controller
    -> Search L1 / Expand graph / Inspect L0 / Compare nodes
    -> reassess evidence sufficiency until Finish or a hard budget
    -> grounded Working Evidence + exact Recent4 -> final answer
```

The model autonomously selects the next tool and stopping point. A deterministic executor validates
memory IDs, causal boundaries, per-tool budgets, and output schemas; it does not choose the tool for the
model. `--controller-mode deterministic` is retained only as a controller ablation.

Memory is isolated by video ID:

- **L0** stores raw frames, timestamps, and CLIP embeddings.
- **L1** stores query-independent block overviews and semantically admitted transitions with L0 links.
- **Working Evidence** is question-local, records retrieval/verification state, and is cleared after the answer.

CLIP change pairs are proposals, not facts. Only a frozen Qwen3-VL assessment classified as a local
object action or state change may enter searchable TransitionMemory. Scene cuts and camera motion remain
navigation metadata; rejected candidates remain auditable but are not retrieved as evidence.

## Project Basis

This project extends the [SimpleStream](https://github.com/EvolvingLMMs-Lab/SimpleStream) recent-window
baseline and keeps its OVO-Bench evaluation conventions. We thank the SimpleStream, Qwen-VL, OVO-Bench,
CLIP, and BGE authors for releasing the code, models, and data used in this training-free study. Their
licenses apply to the corresponding upstream assets.

## Installation

```bash
conda create -n simplestream-qwen3 python=3.10 -y
conda activate simplestream-qwen3
pip install -r requirements.txt
pip install -r requirements-qwen3.txt
```

The main run requires one Qwen3-VL checkpoint, CLIP ViT-L/14, BGE-base-en-v1.5, `accelerate`, and eight
CUDA devices. Passing separate perception and reasoning paths loads two Qwen instances per GPU; add
`--share-model` when memory is insufficient.

## OVO-Bench Data

Use the current annotation from the official [OVO-Bench repository](https://github.com/JoeLeelyf/OVO-Bench)
and the pre-chunked videos from the official
[Hugging Face dataset](https://huggingface.co/datasets/JoeLeelyf/OVO-Bench). The local annotation has
1,640 entries and expands to 3,035 video-question instances. Our primary split contains 631 Backward and
837 Realtime questions (1,468 total) across nine task types.

```bash
pip install -U huggingface_hub hf_xet
mkdir -p data/ovo_bench

curl -fL \
  https://raw.githubusercontent.com/JoeLeelyf/OVO-Bench/main/data/ovo_bench_new.json \
  -o data/ovo_bench/ovo_bench_new.json

for part in aa ab ac ad ae af ag ah ai aj ak al am an ao; do
  hf download JoeLeelyf/OVO-Bench \
    "chunked_videos.tar.part${part}" \
    --repo-type dataset --local-dir data/ovo_bench
done

cat data/ovo_bench/chunked_videos.tar.part{aa,ab,ac,ad,ae,af,ag,ah,ai,aj,ak,al,am,an,ao} \
  | tar -xf - -C data/ovo_bench
```

Expected layout:

```text
data/ovo_bench/
|- ovo_bench_new.json
`- chunked_videos/
   |- 0.mp4
   |- 1.mp4
   `- ...
```

Validate every Backward/Realtime input before launching:

```bash
python - <<'PY'
import json
from pathlib import Path
from ovo_constants import BACKWARD_TASKS, REAL_TIME_TASKS

rows = json.loads(Path("data/ovo_bench/ovo_bench_new.json").read_text())
tasks = set(BACKWARD_TASKS + REAL_TIME_TASKS)
selected = [row for row in rows if row["task"] in tasks]
missing = [row["id"] for row in selected
           if not Path(f"data/ovo_bench/chunked_videos/{row['id']}.mp4").is_file()]
print(f"Backward/Realtime questions: {len(selected)}")
print(f"missing videos: {len(missing)}")
if missing:
    raise SystemExit(f"first missing IDs: {missing[:20]}")
PY
```

OVO-Bench data is not redistributed here. Follow its official license and the licenses of source videos.

## Required Uniform Baseline

The required baseline uses the same Qwen3-VL-8B checkpoint, 32 frames uniformly sampled from the causal
prefix, and one direct QA call. It evaluates only Backward and Realtime:

```bash
nohup env NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  DECORD_EOF_RETRY_MAX=20480 QWEN_EXACT_RECENT_DECODE=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  accelerate launch --num_processes=8 --main_process_port=29730 \
    main_experiments/eval_qwen3vl_ovo.py \
    --model_path /data1/chenjunwei/models/Qwen3-VL-8B-Instruct \
    --anno_path data/ovo_bench/ovo_bench_new.json \
    --chunked_dir data/ovo_bench/chunked_videos \
    --result_dir main_experiments/results/ovo_qwen3vl_8b_uniform32_br_rt \
    --frame_selection uniform --recent_frames_only 32 \
    --chunk_duration 1.0 --fps 1.0 --max_qa_tokens 256 \
    --eval_splits backward,realtime \
  > logs/ovo_qwen3vl_8b_uniform32_br_rt.out 2>&1 &
```

## VeriStream Evaluation

The following is the primary autonomous-agent run. Omit `--share-model` to load independent perception
and reasoning instances on each GPU. The latest semantic index uses schema 6 and must not reuse an older
cache.

```bash
nohup env NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  DECORD_EOF_RETRY_MAX=20480 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  accelerate launch --num_processes=8 --main_process_port=29731 \
    main_experiments/eval_qwen3vl_ovo_budgeted.py \
    --perception-model-path /data1/chenjunwei/models/Qwen3-VL-8B-Instruct \
    --reasoning-model-path /data1/chenjunwei/models/Qwen3-VL-8B-Instruct \
    --anno-path data/ovo_bench/ovo_bench_new.json \
    --chunked-dir data/ovo_bench/chunked_videos \
    --result-dir main_experiments/results/veristream_8b_agent_br_rt \
    --chunk-duration 1.0 --fps 1.0 --recent-frames 4 \
    --history-block-seconds 12 --coverage-frames-per-block 4 \
    --change-peaks-per-block 2 --max-index-frames-per-block 8 \
    --minimum-peak-distance 2.0 --change-spans 1,2,4 \
    --minimum-change-distance 0.05 --change-mad-scale 0.5 \
    --max-index-tokens 2048 --max-qa-tokens 0 \
    --controller-mode llm --visual-verification-policy deterministic \
    --current-lane-decoder exact --current-fast-path-confidence 0.7 \
    --max-tool-actions 4 --max-search-rounds 2 \
    --max-navigation-steps 2 --max-visual-actions 2 \
    --max-history-visual-frames 8 \
    --candidate-metadata-tokens 768 --history-context-tokens 768 \
    --minimum-semantic-similarity 0.25 \
    --minimum-visual-similarity 0.20 --minimum-bm25-score 0.10 \
    --clip-model openai/clip-vit-large-patch14 \
    --text-embedding-model BAAI/bge-base-en-v1.5 \
  > logs/veristream_8b_agent_br_rt.out 2>&1 &
```

`finish_retrieval` is free; the four-action limit applies only to executed Search/Inspect/Expand/Compare
calls. It prevents a malformed controller loop from making evaluation unbounded and defines a measurable
accuracy-cost frontier. Each result stores the full tool trace and stop reason.

Restarting the same command resumes completed rows and compatible schema-6 cache entries. To reuse a
completed index in a new question-time ablation, pass `--index-cache-dir OLD_RESULT_DIR/memory`; the source
is read-only. Change the result directory or pass `--rebuild-memory` only when rebuilding intentionally.

## Scoring And Analysis

The evaluator writes `budgeted_ovo_backward_realtime.json` plus task averages and diagnostics. In this
project, `Total = (Backward + Realtime) / 2`; Forward is not included.

```bash
python main_experiments/compare_budgeted_results.py --help
python main_experiments/eval_veristream_planner_routing.py --help
python main_experiments/export_transition_audit.py --help
python main_experiments/score_transition_audit.py --help
```

These scripts are offline analysis tools, not runtime dependencies. Removing them does not change an
eight-GPU evaluation, but it removes paired significance testing, routing calibration, and semantic-gate
audit support.

## Verification

```bash
python -m unittest discover -s tests -q
python -m py_compile lib/veristream_budgeted.py \
  main_experiments/eval_qwen3vl_ovo_budgeted.py
```

See [`实验报告.md`](实验报告.md) for the motivation, method, results, ablations, and limitations.

## Core Files

```text
lib/veristream_budgeted.py                    VeriStream index, retrieval, tools, and agent
lib/qwen_exact_recent_decoder.py              exact Recent4 current decoder
main_experiments/eval_qwen3vl_ovo.py          uniform/direct baselines
main_experiments/eval_qwen3vl_ovo_budgeted.py OVO Backward/Realtime agent evaluation
docs/veristream/gpt_image2_method_figure_prompt.md
tests/test_veristream_budgeted.py              logic and regression tests
```
