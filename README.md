# VeriStream

Training-free long-video understanding with causal evidence memory and decoupled perception/reasoning.

VeriStream uses frozen Qwen3-VL models. A perception role builds timestamped evidence cards, a
reasoning role retrieves and verifies historical evidence, and the final answer combines verified
memory with the causal recent visual window. The orchestration, memory state machine, and tools are
implemented locally; no video-agent framework is required.

## Method

- Causal access: only frames before the question timestamp are exposed.
- Hierarchical memory: raw frame pointers, observations, events, and a question working set.
- Tools: `search_memory`, `inspect_segment`, `verify_evidence`, and `compare_segments`.
- Hybrid-4: verified historical evidence plus four recent raw frames.
- Training-free: model weights are frozen and no fine-tuning is used.

The design, failure analysis, results, and reproduction settings are summarized below so the code
repository remains self-contained.

## Project basis and acknowledgements

This project is developed on top of the [SimpleStream](https://github.com/EvolvingLMMs-Lab/SimpleStream)
recent-window baseline. We thank the SimpleStream authors for the baseline implementation and OVO-Bench
evaluation pipeline. We also thank the Qwen-VL, OVO-Bench, and StreamingBench authors for releasing the
models and benchmarks used here.

## Method summary

For each causal video prefix, the perception role builds a question-independent evidence index. Each
observation keeps its time interval and raw chunk pointers. The reasoning role retrieves candidate
observations with `search_memory`, requests local visual evidence with `inspect_segment`, and promotes
only supported candidates through `verify_evidence`; conflicting candidates are quarantined. The final
Hybrid-4 answer receives verified historical memory plus the four most recent raw frames. This preserves
long-range evidence without removing the current visual state. `lib/veristream.py` contains the shared
memory status model and the single-role StreamingBench compatibility path; `lib/veristream_dual_role.py`
contains the current evidence index and dual-role controller.

## Installation

Install PyTorch and torchvision for your CUDA runtime first, then install the Qwen3-VL stack:

```bash
conda create -n simplestream-qwen3 python=3.10 -y
conda activate simplestream-qwen3
pip install -r requirements-qwen3.txt
```

The repository does not include model weights or datasets. Install the Hugging Face CLI for dataset downloads:

```bash
pip install -U huggingface_hub
```

### OVO-Bench

Download the official annotation file and the pre-chunked videos. The video archive is split into
15 parts and is approximately 144 GB in total.

```text
data/ovo_bench/ovo_bench_new.json
data/ovo_bench/chunked_videos/
```

```bash
mkdir -p data/ovo_bench
curl -L https://raw.githubusercontent.com/JoeLeelyf/OVO-Bench/main/data/ovo_bench_new.json \
  -o data/ovo_bench/ovo_bench_new.json

for part in aa ab ac ad ae af ag ah ai aj ak al am an ao; do
  hf download JoeLeelyf/OVO-Bench \
    "chunked_videos.tar.part${part}" \
    --repo-type dataset --local-dir data/ovo_bench
done

cat data/ovo_bench/chunked_videos.tar.part{aa,ab,ac,ad,ae,af,ag,ah,ai,aj,ak,al,am,an,ao} \
  > data/ovo_bench/chunked_videos.tar
tar -xf data/ovo_bench/chunked_videos.tar -C data/ovo_bench
rm data/ovo_bench/chunked_videos.tar data/ovo_bench/chunked_videos.tar.part*
```

### StreamingBench

The official release is hosted at `mjuicem/StreamingBench`. Download all archives, then unpack
them into the official category directories:

```bash
mkdir -p data/streamingbench/raw
hf download mjuicem/StreamingBench \
  --repo-type dataset --local-dir data/streamingbench/raw

mkdir -p data/streamingbench/{real,omni,sqa,proactive}
for archive in data/streamingbench/raw/Real-Time\ Visual\ Understanding_*.zip; do
  unzip -q "$archive" -d data/streamingbench/real
done
for archive in data/streamingbench/raw/Proactive\ Output_*.zip; do
  unzip -q "$archive" -d data/streamingbench/proactive
done
for archive in data/streamingbench/raw/Sequential\ Question\ Answering_*.zip; do
  unzip -q "$archive" -d data/streamingbench/sqa
done
for archive in data/streamingbench/raw/*.zip; do
  case "$archive" in
    *"Real-Time Visual Understanding"*|*"Proactive Output"*|*"Sequential Question Answering"*) ;;
    *) unzip -q "$archive" -d data/streamingbench/omni ;;
  esac
done
```

The official CSV files and preprocessing details are maintained at
`https://github.com/THUNLP-MT/StreamingBench`. The evaluator in this repository expects the
preprocessed `questions_real.json` and flat `videos/` directory.

## Tests

```bash
python -m unittest discover -s tests -v
```

## OVO-Bench baseline

The baseline evaluates Backward, Realtime, and Forward with a frozen Qwen3-VL model:

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 \
  main_experiments/eval_qwen3vl_ovo.py \
  --model_path /path/to/Qwen3-VL-8B-Instruct \
  --anno_path data/ovo_bench/ovo_bench_new.json \
  --chunked_dir data/ovo_bench/chunked_videos \
  --result_dir main_experiments/results/ovo_qwen3vl_recent4 \
  --frame_selection recent --recent_frames_only 4 \
  --chunk_duration 1.0 --fps 1.0 --max_qa_tokens 256
```

The same entry point reproduces the SimpleStream frame and text-memory baselines by changing only
`--frame_selection` and the frame budget:

| Experiment | Arguments |
| --- | --- |
| Recent4/16/32 | `--frame_selection recent --recent_frames_only 4/16/32` |
| Uniform16/32 | `--frame_selection uniform --recent_frames_only 16/32` |
| CLIP-TopK16/32 | `--frame_selection clip_topk --recent_frames_only 16/32 --clip_model_path openai/clip-vit-large-patch14` |
| Recent4 + Uniform16 | `--frame_selection recent_uniform --recent_frames_only 4 --supplemental_frames 16` |
| Recent4 + CLIP-TopK16 | `--frame_selection recent_clip_topk --recent_frames_only 4 --supplemental_frames 16 --clip_model_path openai/clip-vit-large-patch14` |
| Action-fact + Uniform16 | `--frame_selection recent_memory_uniform --recent_frames_only 4 --supplemental_frames 16` |
| Action-fact + CLIP-TopK16 | `--frame_selection recent_memory_clip_topk --recent_frames_only 4 --supplemental_frames 16 --clip_model_path openai/clip-vit-large-patch14` |
| State-aware + Uniform16 | `--frame_selection recent_state_memory_uniform_v4 --recent_frames_only 4 --supplemental_frames 16` |
| State-aware + CLIP-TopK16 | `--frame_selection recent_state_memory_clip_topk_v4 --recent_frames_only 4 --supplemental_frames 16 --clip_model_path openai/clip-vit-large-patch14` |
| Recent4 + Uniform28 | `--frame_selection recent_uniform --recent_frames_only 4 --supplemental_frames 28` |
| Recent4 + CLIP-TopK28 | `--frame_selection recent_clip_topk --recent_frames_only 4 --supplemental_frames 28 --clip_model_path openai/clip-vit-large-patch14` |

Append these shared arguments to every baseline command:

```bash
--anno_path data/ovo_bench/ovo_bench_new.json \
--chunked_dir data/ovo_bench/chunked_videos \
--chunk_duration 1.0 --fps 1.0 --max_qa_tokens 256
```

## VeriStream Hybrid-4

Backward and Realtime use the dual-role entry point:

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 \
  main_experiments/eval_qwen3vl_ovo_dual_role.py \
  --perception-model-path /path/to/Qwen3-VL-8B-Instruct \
  --reasoning-model-path /path/to/Qwen3-VL-8B-Instruct \
  --result-dir main_experiments/results/veristream_dual_role_hybrid4_br_rt \
  --chunk-duration 1.0 --fps 1.0 --coarse-stride 12 \
  --max-frames-per-observation 8 --max-actions 4 \
  --max-qa-tokens 256 --final-recent-frames 4
```

Forward uses `main_experiments/eval_qwen3vl_ovo_dual_role_forward.py` with the same model and
configuration. Keep the same `--result-dir` to resume from rank checkpoints.

## Recorded OVO-Bench results

All values below are percentages from the same OVO-Bench split scoring protocol. `Total` is the
arithmetic mean of the three split averages; the text-only dual-role ablation has no Forward run and
therefore uses a two-split mean marked with `*`.

| Model / method | Backward | Realtime | Forward | Total |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-VL-2B Recent4 | 53.56 | 74.23 | 42.67 | 56.82 |
| Qwen3-VL-2B Recent4 + CLIP-TopK16 | 52.73 | 69.52 | 43.12 | 55.13 |
| Qwen3-VL-2B action-fact memory + Uniform16 | 56.91 | 73.68 | 38.84 | 56.48 |
| Qwen3-VL-8B Recent4 | 53.92 | 81.47 | 39.40 | 58.26 |
| Qwen3-VL-8B Recent16 | 55.06 | 77.80 | 43.94 | 58.93 |
| Qwen3-VL-8B Recent32 | 57.35 | 76.57 | 46.08 | 60.00 |
| Qwen3-VL-8B action-fact memory + Uniform16 | 62.09 | 79.65 | 39.04 | 60.26 |
| VeriStream text-only | 55.05 | 44.91 | -- | 49.98* |
| VeriStream Hybrid-4 | 57.60 | 79.18 | 39.07 | 58.62 |

The main finding is a memory-perception trade-off: larger raw windows improve some historical or
forward questions but reduce real-time focus. Text memory improves historical recall on 8B while
preserving recent frames. Hybrid-4 restores real-time perception over text-only dual-role inference
by supplying raw recent vision, while retaining verified historical evidence.

## Code layout

```text
lib/veristream.py                         # single-role memory and tool state machine
lib/veristream_dual_role.py               # evidence index and dual-role orchestration
lib/clip_topk_selector.py                 # CLIP semantic frame selection
lib/recent_window_eval_qwen3.py           # Qwen3-VL decoding and evaluation helpers
main_experiments/eval_qwen3vl_ovo.py      # recent/uniform/CLIP/text-memory baseline
main_experiments/eval_qwen3vl_ovo_dual_role.py
main_experiments/eval_qwen3vl_ovo_dual_role_forward.py
scoring/score_ovo_bench.py
tests/                                     # deterministic memory and tool tests
```

## License and data

Please follow the licenses of Qwen3-VL, OVO-Bench, StreamingBench, and all upstream dependencies.
Datasets, videos, model weights, logs, and generated result files are intentionally excluded from
this repository.
