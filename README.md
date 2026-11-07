# DFlash-Qwen3-8B

**Training and evaluating a custom DFlash speculative-decoding model for Qwen3-8B.**

DFlash-Qwen3-8B is an experimental inference-engineering project. It trains a DFlash draft model for `Qwen/Qwen3-8B`, serves the checkpoint with vLLM, and measures whether speculative decoding accepts enough draft tokens to improve serving performance.

> **Current result:** The latest recorded checkpoint loads in vLLM 0.31.0 and reaches a mean acceptance length of **1.50**. This demonstrates that the draft is functioning, but it is **not evidence of a serving speedup**. A target-only baseline and goodput measurements have not yet been completed.

> **Important:** The completed runs below do not implement the locked five-auxiliary-layer v0.1 recipe exactly. Run B trained with four feature layers. The intended five-layer recipe remains a follow-up.

## Contents

- [Project goals](#project-goals)
- [Current status](#current-status)
- [What actually ran](#what-actually-ran)
- [Locked v0.1 configuration](#locked-v01-configuration)
- [Training results](#training-results)
- [Serving measurement](#serving-measurement)
- [Serving with vLLM](#serving-with-vllm)
- [Benchmarking methodology](#benchmarking-methodology)
- [Repository structure](#repository-structure)
- [Reproducibility and environment](#reproducibility-and-environment)
- [Limitations and next steps](#limitations-and-next-steps)

## Project goals

1. Train a custom DFlash speculator for `Qwen/Qwen3-8B`.
2. Serve the trained checkpoint with vLLM speculative decoding.
3. Measure acceptance length and per-position acceptance, not only validation loss.
4. Compare target-only generation against DFlash-assisted generation.
5. Measure TTFT, TPOT, throughput, and SLO goodput once the draft is strong enough to justify a performance comparison.

## Current status

| Component | Status | Notes |
|---|---|---|
| Project structure and contract | Done | Repository and intended recipe documented |
| Prepared data and vocabulary mappings | Done for recorded runs | Includes token-frequency file and `d2t` / `t2d` mappings |
| Offline hidden-state extraction | Done | Used for recorded training runs |
| Three-epoch training | Done | Two runs completed |
| Local checkpoint loading in vLLM | Done | Tested with vLLM 0.31.0 |
| Acceptance measurement | Partial | One short serving measurement; latest mean acceptance length 1.50 |
| Target-only baseline | Not done | No comparable baseline yet |
| Repeated acceptance runs | Not done | Current serving result is a short run |
| TTFT, TPOT, throughput, and goodput | Not done | No performance claim can be made |
| Speculative-token sweep | Not done | Planned |
| Locked five-layer retraining | Not done | Requires six extraction slots including final verifier state, then training on five auxiliary layers |
| Kill switch | Optional / not done | Defer until core evaluation is complete |
| Hugging Face publication | Not done | Publish only after intended layer configuration and data are validated |

## What actually ran

The locked recipe is the intended baseline. It is not the exact recipe that produced the latest checkpoint. Deviations are recorded here rather than silently changing the project contract.

| Setting | Locked v0.1 | Run A (first completed run) | Run B (current best) |
|---|---|---|---|
| Prepared data | Full sequences | `prepared-2000`, 165 rows after filtering | `prepared-500`, hidden states for 674 rows |
| Responses | On-policy UltraChat | Existing 193-row JSONL | `ultrachat-qwen3-8b-500.jsonl`, 500 successful responses; all truncated |
| Sequence length | 8192 | 2000 | 2000 |
| Maximum anchors | 3072 | 512 | 512 |
| Epochs | 3 | 3 | 3 |
| Auxiliary layers requested | `[2, 10, 18, 26, 34]` | `[2, 10, 18, 26, 34]` | Extracted tensor shape `[seq, 5, 4096]` |
| Auxiliary layers reaching the draft projection | 5 | 5 on the earlier small run | 4: `[2, 10, 18, 26]` |
| Hidden-state workflow | On-the-fly or offline | On-the-fly; context errors until sequence length was reduced | Offline, then `--on-missing raise` |
| Checkpoint | — | `artifacts/training/dflash-qwen3-8b-full` | `artifacts/training/dflash-qwen3-8b-500` |

### Why Run B used four feature layers

Each saved Run B tensor had shape `(sequence_length, 5, 4096)`. This Speculators build drops the last layer before the draft projection, so the projection received four feature layers:

```text
4 × 4096 = 16384
```

The five-layer draft projection expects:

```text
5 × 4096 = 20480
```

Training with five `target_layer_ids` against those five-slot files therefore produces a matrix-dimension mismatch.

The Red Hat recipe used as a reference avoids this by extracting six IDs—`[2, 10, 18, 26, 34, 36]`—and training with the first five auxiliary IDs. Run B did not use that six-slot extraction contract; it trained on the four layers that matched the files on disk.

**Do not describe Run B as a completed five-auxiliary-layer implementation.**

## Locked v0.1 configuration

Treat this as the intended baseline. Change it only in a versioned follow-up. Run B did not implement this table exactly.

| Parameter | Locked value |
|---|---|
| Target model | `Qwen/Qwen3-8B` |
| Speculator | DFlash |
| Auxiliary hidden layers | `[2, 10, 18, 26, 34]` |
| DFlash layers | 5 |
| Draft vocabulary | 32,000 |
| Block size | 8 |
| Speculative tokens | 7 |
| Maximum anchors | 3,072 |
| Training sequence length | 8,192 |
| Training epochs | 3 |
| Learning rate | `6e-4` |
| Optimizer | Muon (`muon_lr=6e-4`, momentum `0.95`, weight decay `0.1`, 5 Newton–Schulz steps, `match_rms_adamw`) plus AdamW on remaining parameters |
| Loss | Cross-entropy with D-PACE, `dpace_alpha=0.5` |
| Scheduler | Cosine |
| Random seed | 42 |
| Training precision | BF16 |

## Training results

### Run A — 165 filtered rows

Best checkpoint was recorded at epoch 2. This is the run behind the first serving measurement (mean acceptance length 1.21).

| Metric | Result |
|---|---:|
| Checkpoint | `artifacts/training/dflash-qwen3-8b-full` |
| Best epoch | 2 |
| Global step | 138 |
| Validation loss | 1.892 |
| Validation full accuracy | 7.16% |
| Validation EAL | 1.208 |

Per-position validation accuracy:

| Position | Accuracy |
|---:|---:|
| 1 | 18.0% |
| 2 | 8.6% |
| 3 | 5.7% |
| 4 | 5.1% |
| 5 | 4.5% |
| 6 | 4.2% |
| 7 | 4.0% |

A later pass on the same small set reached validation EAL 1.314 (loss 1.845, position-1 accuracy 24.9%). Serving of that checkpoint reported mean acceptance length **1.21**, position-1 acceptance **21.2%**, and draft acceptance rate **3.0%** (11 accepted / 364 drafted).

### Run B — 500 regenerated conversations

This is the current best recorded checkpoint. It was saved at epoch 2. The feature layers passed to the draft were `[2, 10, 18, 26]`.

| Metric | Result |
|---|---:|
| Checkpoint | `artifacts/training/dflash-qwen3-8b-500/checkpoint_best` |
| Best epoch | 2 |
| Validation loss | 1.709 |
| Validation full accuracy | 13.2% |
| Position 1 accuracy | 34.0% |
| Position 2 accuracy | 20.6% |
| Position 3 accuracy | 12.8% |
| Position 4 accuracy | 8.7% |
| Position 5 accuracy | 6.6% |
| Position 6 accuracy | 5.0% |
| Position 7 accuracy | 4.2% |
| Validation EAL | **1.492** |

Every reported validation number improved versus Run A. However, EAL 1.49 is still too low to claim a serving speedup.

## Serving measurement

Recorded setup: vLLM 0.31.0, Qwen3-8B verifier, Run B checkpoint, `num_speculative_tokens=7`, `max_model_len=4096`, temperature 0. This was one short batch of completion requests, not a benchmark.

| Metric | Run A | Run B |
|---|---:|---:|
| Mean acceptance length | 1.21 | **1.50** |
| Position 1 acceptance | 21.2% | **40.5%** |
| Position 2 acceptance | 0% | **9.5%** |
| Positions 3–7 | 0% | 0% |
| Draft acceptance rate | 3.0% | **7.1%** |
| Accepted / drafted tokens | 11 / 364 | 21 / 294 |

Serving mean acceptance length is consistent with validation EAL (1.50 versus 1.492). The draft behaves as trained, but this does not establish a speedup: no target-only baseline was collected, and most drafted tokens are still rejected.

An older single request on an earlier checkpoint generated 100 tokens in 6.85 seconds (14.59 tokens/s). **Do not cite this as a speedup.**

## Serving with vLLM

Recorded Run B environment:

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 4500 Blackwell, 32 GB |
| Python | 3.12 |
| PyTorch | `2.13.0+cu130` |
| vLLM | `0.31.0` |
| Speculators | `0.9.0a72` (training environment that produced the checkpoint) |
| Project path | `/workspace/dflash-qwen3-8b-src` |
| Training environment | `/workspace/dflash-qwen3-8b/.venv-train` |
| Serving environment | `/workspace/dflash-qwen3-8b/.venv-serve` |

Start the server with an absolute checkpoint path. A relative path can fail if the shell is not in the project directory. `checkpoint_best` is a symlink to the best epoch directory and must contain `config.json` and `model.safetensors`.

```bash
source /workspace/dflash-qwen3-8b/.venv-serve/bin/activate

vllm serve Qwen/Qwen3-8B \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --speculative-config '{"method":"dflash","model":"/workspace/dflash-qwen3-8b-src/artifacts/training/dflash-qwen3-8b-500/checkpoint_best","num_speculative_tokens":7}'
```

### Smoke request

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "prompt": "Explain speculative decoding in one paragraph.",
    "max_tokens": 64,
    "temperature": 0
  }'
```

Acceptance metrics are not included in the HTTP response. On this vLLM build they are also absent from `/metrics`. Read the server log for a line similar to:

```text
SpecDecoding metrics: Mean acceptance length: ...
```

## Benchmarking methodology

Do not claim a speedup until a target-only run exists under the same GPU, `max_model_len`, sampling settings, prompts, and concurrency.

1. **Target-only baseline:** Serve `Qwen/Qwen3-8B` without `--speculative-config`.
2. **Acceptance:** Record mean acceptance length, per-position acceptance, and drafted versus accepted tokens. Use the server's `SpecDecoding metrics` log line.
3. **Latency and throughput:** Measure TTFT, TPOT, end-to-end latency, output tokens/s, requests/s, and peak GPU memory. Report medians and spread across repeated runs.
4. **Speedup:** Calculate DFlash throughput divided by target-only throughput. A value above 1 is a speedup only for that tested workload.
5. **Goodput:** Define TTFT and TPOT limits before testing. Goodput is requests meeting both limits divided by elapsed time.
6. **Speculative-token sweep:** Change only `num_speculative_tokens` (7, then 5 and 3). Record acceptance and latency for each setting.

For the planned concurrency matrix, test `C = 1, 4, 8, 16` with speculation on and off, keeping workload and SLO definitions fixed.

## Repository structure

The recorded project layout is:

```text
dflash-qwen3-8b-src/
├── configs/
│   └── train_dflash_qwen3_8b.yaml
├── scripts/
│   └── prepare_dflash_training.py
├── artifacts/
│   ├── data-final/
│   │   └── prepared-500/          # Arrow data, token_freq.pt, d2t, t2d, hidden_states
│   └── training/
│       ├── dflash-qwen3-8b-500/    # current recorded checkpoint
│       └── dflash-qwen3-8b-full/  # earlier 165-row checkpoint
├── logs/
└── README.md
```

Large checkpoints and hidden-state caches should stay out of Git. The training machine's `/workspace` volume was 49 GB and filled while saving optimizer state and hidden states. Remove only verified-unused epoch directories and old prepared datasets before another run; retain the data and checkpoints needed for reproducibility.

## Reproducibility and environment

### Run B training command

The following command records the Run B setup. It is not the locked v0.1 five-layer recipe.

```bash
source /workspace/dflash-qwen3-8b/.venv-train/bin/activate

python -m speculators.train \
  --config artifacts/upstream-dflash-train.yaml \
  --data-path artifacts/data-final/prepared-500 \
  --total-seq-len 2000 \
  --max-anchors 512 \
  --gradient-checkpointing \
  --num-workers 0 \
  --on-missing raise \
  --target-layer-ids 2 10 18 26 \
  --save-path artifacts/training/dflash-qwen3-8b-500 \
  --epochs 3
```

Hidden states must already exist under `artifacts/data-final/prepared-500/hidden_states`. Stop the vLLM extractor before training. Training and a vLLM extractor configured at `0.55` GPU memory utilization do not fit together on this 32 GB GPU.

For every later benchmark row, record:

- Git commit
- Target model revision
- Checkpoint path and revision
- GPU
- Python, PyTorch, CUDA, vLLM, and Speculators versions
- Prompt set
- Warm-up procedure
- Number of repetitions
- Concurrency
- Output length

## Limitations and next steps

Current limitations:

- Acceptance improved as the dataset grew; no controlled tuning experiment has established why.
- All 500 regenerated responses were truncated, limiting draft quality.
- Run B's draft received four verifier feature layers, not the intended five.
- Positions 3–7 acceptance were zero in the recorded serving measurement, so `K=7` performs mostly rejected draft work.
- No target-only baseline exists; therefore, no speedup claim is supported.
- The recorded serving measurement is short and has not been repeated.

Recommended next steps, in order:

1. Record a target-only baseline with the same prompts and `max_model_len`.
2. Regenerate a few thousand responses with a higher `max_tokens` limit so responses are not all truncated.
3. Extract hidden states with IDs `[2, 10, 18, 26, 34, 36]`, then train with `--target-layer-ids 2 10 18 26 34`.
4. Re-measure mean acceptance length. If it remains near 1.5, investigate data quality and feature alignment before claiming progress. Continue to goodput testing only when acceptance is materially better (the project target is around 2.0 or higher).
5. Publish the checkpoint only after the layer list and dataset match the intended recipe.

## Success criteria

The project should be considered reproducible when another engineer can:

- Recreate the training data and hidden-state cache using documented commands.
- Train the intended five-auxiliary-layer checkpoint.
- Load and serve that checkpoint in vLLM.
- Reproduce acceptance measurements.
- Compare target-only and speculative serving under the same workload.
- Inspect published latency, throughput, and goodput results without relying on unsupported speedup claims.
