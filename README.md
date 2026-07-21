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

The complete design, failure analysis, experiment tables, and reproduction commands are in:

- [实验报告](docs/veristream/实验报告.md)
- [设计说明文档](docs/veristream/设计说明文档.md)
- [复现指南](docs/veristream/复现指南.md)
- [LaTeX 结果表](docs/veristream/ovo_results_tables.tex)

## Installation

Install PyTorch and torchvision for your CUDA runtime first, then install the Qwen3-VL stack:

```bash
conda create -n simplestream-qwen3 python=3.10 -y
conda activate simplestream-qwen3
pip install -r requirements-qwen3.txt
```

The repository does not include model weights or datasets. Put OVO-Bench files at:

```text
data/ovo_bench/ovo_bench_new.json
data/ovo_bench/chunked_videos/
```

StreamingBench paths are documented in [复现指南](docs/veristream/复现指南.md).

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

## Code layout

```text
lib/veristream.py                         # single-role memory and tool state machine
lib/veristream_dual_role.py               # evidence index and dual-role orchestration
lib/recent_window_eval_qwen3.py           # Qwen3-VL decoding and evaluation helpers
main_experiments/eval_qwen3vl_ovo.py      # recent/uniform/CLIP/text-memory baseline
main_experiments/eval_qwen3vl_ovo_dual_role.py
main_experiments/eval_qwen3vl_ovo_dual_role_forward.py
main_experiments/build_veristream_evidence_index.py
main_experiments/query_veristream_dual_role.py
scoring/score_ovo_bench.py
tests/                                     # deterministic memory and tool tests
```

## License and data

Please follow the licenses of Qwen3-VL, OVO-Bench, StreamingBench, and all upstream dependencies.
Datasets, videos, model weights, logs, and generated result files are intentionally excluded from
this repository.
