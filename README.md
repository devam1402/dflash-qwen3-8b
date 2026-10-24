# DFlash-Qwen3-8B

**Training and evaluating a custom DFlash speculative decoding model for Qwen3-8B.**

DFlash-Qwen3-8B is an experimental inference-engineering project exploring how a trained speculative draft model can accelerate autoregressive LLM generation. The project covers draft-model training, checkpoint management, vLLM serving, acceptance analysis, latency and throughput measurement, SLO-aware goodput, and runtime controls.

The objective is to build a reproducible workflow that measures whether speculative decoding improves real serving performance—not merely whether the draft model trains successfully.

> **Project status:** Custom training completed, checkpoint available locally, and initial vLLM inference verified. Comparative performance benchmarks and public checkpoint publication remain in progress.

## Table of contents

- [Goals](#goals)
- [Architecture](#architecture)
- [Locked v0.1 configuration](#locked-v01-configuration)
- [Implementation status](#implementation-status)
- [Training results](#training-results)
- [Serving with vLLM](#serving-with-vllm)
- [Benchmarking methodology](#benchmarking-methodology)
- [Repository structure](#repository-structure)
- [Reproducibility](#reproducibility)
- [Limitations and next steps](#limitations-and-next-steps)

## Goals

- Train a custom DFlash speculator for Qwen3-8B.
- Integrate the draft model with vLLM speculative decoding.
- Measure speculative acceptance and effective accepted tokens.
- Compare target-only generation against DFlash-assisted generation.
- Evaluate time to first token (TTFT), time per output token (TPOT), and end-to-end throughput.
- Measure goodput under explicit service-level objectives (SLOs).
- Investigate the effect of speculative token count and workload characteristics.
- Provide a runtime mechanism to disable speculative decoding.
- Publish reproducible configurations, benchmark scripts, and results.

## Architecture

```text
                   Training pipeline
                          |
              +-----------+-----------+
              |                       |
          Qwen3-8B                Training data
              |                       |
              +---- Hidden states ----+
                          |
                          v
                  DFlash training
                          |
                          v
                   Trained draft
                          |
                          v
                  Checkpoint artifact
                          |
                  Hugging Face Hub
                  (planned publication)
                          |
                          v
                    vLLM serving
                          |
                 +--------+--------+
                 |                 |
                 v                 v
             DFlash draft      Qwen3-8B
                 |              verifier
                 +-------+---------+
                         |
                         v
                 Accepted/rejected
                  speculative tokens
                         |
                         v
                 Runtime measurements
                         |
              +----------+----------+
              |          |          |
              v          v          v
          Acceptance   Latency    Throughput
              |          |          |
              +----------+----------+
                         |
                         v
                  SLO-aware goodput
```

The draft model proposes multiple tokens in one speculative block. The target model verifies the proposal, allowing compatible tokens to be accepted while preserving the target model's generation distribution under the supported speculative decoding implementation.

The main question is whether the additional draft computation and verification reduce the cost of producing output tokens on the target hardware.

## Locked v0.1 configuration

The following recipe defines the project's initial experiment. Treat it as the baseline configuration; change parameters only in explicitly versioned follow-up experiments.

| Parameter | Value |
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
| Scheduler | Cosine |
| Random seed | 42 |
| Training precision | BF16 |

**Important:** Training sequence length and serving context length are different settings. The initial serving experiment used a maximum model length of 2,048 because the available GPU memory was constrained.

## Implementation status

- [x] Repository structure and project contract
- [x] Locked v0.1 configuration
- [x] Dependency pin location
- [x] Result schema
- [x] CPU validation
- [x] Three-epoch training run
- [x] Local DFlash checkpoint
- [x] vLLM serving smoke test
- [x] Initial 100-token generation timing
- [ ] Repeated benchmark runs and variability analysis
- [ ] Target-only baseline under matching conditions
- [ ] Speculative acceptance benchmark
- [ ] TTFT and TPOT measurements
- [ ] SLO and goodput benchmark
- [ ] Speculative token-count sweep
- [ ] Runtime kill-switch validation
- [ ] Hugging Face checkpoint publication
- [ ] Public benchmark report

## Training results

The initial training run completed three epochs, with the best checkpoint recorded at epoch 2.

| Metric | Observed result |
|---|---:|
| Best checkpoint | `checkpoint_best` |
| Best checkpoint epoch | 2 |
| Global training step | 138 |
| Validation loss | 1.8922 |
| Validation full accuracy | 7.16% |
| Validation effective accepted length (EAL) | 1.2083 |

Per-position validation accuracy:

| Draft position | Accuracy |
|---:|---:|
| 1 | 18.03% |
| 2 | 8.59% |
| 3 | 5.66% |
| 4 | 5.08% |
| 5 | 4.47% |
| 6 | 4.18% |
| 7 | 4.02% |

These are initial validation results, not evidence of an inference speedup. Effective accepted length is a useful draft-quality indicator, but actual serving performance depends on verification cost, batch size, context length, memory use, and workload.

The low full-block accuracy also makes acceptance analysis and further training evaluation important next steps.

### Initial inference measurement

The trained draft checkpoint was loaded into vLLM, and a chat-completion request successfully generated 100 output tokens.

| Measurement | Initial result |
|---|---:|
| Output tokens | 100 |
| End-to-end elapsed time | 6.85 seconds |
| End-to-end throughput | 14.59 tokens/second |
| Finish reason | `length` |

This is a **single preliminary measurement**, not a statistically stable benchmark. A matching target-only measurement is required before claiming a speedup.

## Serving with vLLM

### Environment

The initial experiment used:

- GPU: NVIDIA L4, approximately 23 GB VRAM.
- Python: 3.12.3.
- PyTorch: 2.13.0+cu130.
- vLLM: 0.31.0.
- Speculators: 0.9.0a72.
- Target model: `Qwen/Qwen3-8B`.
- Draft model: locally trained DFlash checkpoint.

These versions describe the recorded experiment environment. Pin and verify compatible versions before reproducing the run.

### Start the speculative decoding server

Activate the project environment and set the project directory:

```bash
source /root/dflash-qwen3-8b-venv/bin/activate
cd /workspace/keen-amber-bison/dflash-qwen3-8b
```

Start vLLM using the trained checkpoint:

```bash
vllm serve "$PWD/artifacts/training/dflash-qwen3-8b-full/checkpoint_best" \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.95
```

The checkpoint configuration identifies the DFlash draft architecture and its Qwen3-8B target. The initial run successfully loaded the target and draft models with seven speculative tokens.

The server runs in the foreground. Wait for the API server to report that it is listening before sending requests.

### Test the OpenAI-compatible API

Run this in a second terminal on the same machine:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/keen-amber-bison/dflash-qwen3-8b/artifacts/training/dflash-qwen3-8b-full/checkpoint_best",
    "messages": [
      {
        "role": "user",
        "content": "Explain speculative decoding in three simple sentences."
      }
    ],
    "max_tokens": 100,
    "temperature": 0
  }'
```

The `model` field must match the model ID exposed by the running server. In the initial experiment, this was the full checkpoint path.

For longer contexts, check GPU memory before increasing `--max-model-len`. The original 40,960-token context configuration failed to initialize because there was insufficient KV-cache memory.

## Benchmarking methodology

The benchmark suite should distinguish model quality, decoding efficiency, and serving performance.

### 1. Target-only baseline

Serve `Qwen/Qwen3-8B` without speculative decoding, on the same GPU and with comparable serving settings.

Use identical prompts, output lengths, sampling settings, and request concurrency. Warm up the server before collecting measurements.

### 2. Speculative acceptance

Record:

- Draft tokens proposed per verification step.
- Tokens accepted per verification step.
- Acceptance rate by draft position.
- Effective accepted length.
- Verification and draft-model overhead.

Use the implementation's reported acceptance statistics where available. Do not infer accepted tokens from generated text alone.

### 3. Latency and throughput

Measure:

| Metric | Meaning |
|---|---|
| TTFT | Time from request submission until the first output token |
| TPOT | Average time per output token after the first token |
| End-to-end latency | Total request duration |
| Output throughput | Generated tokens divided by elapsed time |
| Request throughput | Completed requests per second |
| Peak GPU memory | Maximum observed GPU memory consumption |

Report warm-up policy, prompt length, output length, concurrency, repetitions, and summary statistics. Compare medians and variability across repeated runs rather than relying on a single request.

### 4. Speedup

Calculate throughput speedup using matching baseline and speculative measurements:

`Speedup = DFlash throughput / Target-only throughput`

A value above 1 indicates higher throughput for that particular tested workload. Report latency speedup separately, using matching latency measurements.

### 5. SLO-aware goodput

Define the workload's latency objectives before testing. For example, a request counts as successful only when it completes within the chosen TTFT and TPOT limits.

`Goodput = Requests meeting all defined SLOs / Total elapsed time`

Also report the fraction of requests meeting the SLOs. A faster average is not necessarily better if tail latency or failure rates worsen.

### 6. Speculative token-count sweep

Evaluate supported speculative token counts around the locked seven-token configuration. Record acceptance, throughput, latency, and memory for each configuration.

Change one experimental variable at a time and preserve the baseline results.

## Runtime kill switch

The project intends to provide a way to disable speculative decoding without changing the target model or application interface.

The desired comparison is:

- **Enabled:** target model plus DFlash draft.
- **Disabled:** target model without speculative decoding.

The switch must be implemented and tested before this feature is considered complete. Do not assume that a configuration change automatically provides a safe runtime toggle.

## Repository structure

The following is the intended organization; actual files may vary as implementation progresses.

```text
dflash-qwen3-8b/
├── configs/
│   └── training configuration
├── scripts/
│   ├── launch-hidden-state-vllm.sh
│   └── recover-training.sh
├── tests/
├── results/
│   └── benchmark summaries
├── artifacts/             # Generated files; excluded from Git
├── requirements.txt
├── README.md
└── .gitignore
```

Keep source code, configuration, evaluation scripts, small result summaries, and documentation in Git. Store large model checkpoints and generated datasets outside normal Git history.

## Checkpoint publication

The trained checkpoint currently exists as a local artifact. Hugging Face publication is planned, and the checkpoint revision has not yet been pinned.

Before publishing:

1. Validate the checkpoint and its configuration.
2. Document the target-model dependency and compatible software versions.
3. Upload the required model files and configuration.
4. Verify that the published checkpoint can be downloaded and loaded.
5. Record the immutable Hugging Face revision in the repository.

Do not use the local checkpoint path as though it were a public model identifier.

## Reproducibility

Every benchmark result should record:

- Git commit and configuration version.
- Target model revision.
- Draft checkpoint path or immutable Hub revision.
- GPU model and available VRAM.
- Python, PyTorch, CUDA, vLLM, and Speculators versions.
- Prompt dataset and request parameters.
- Warm-up count, repetitions, concurrency, and output length.
- Acceptance, latency, throughput, and memory measurements.
- Any errors, retries, or excluded runs.

Keep raw results separate from summaries. Do not overwrite previous benchmark results when changing configurations.

## Limitations and next steps

The initial experiment demonstrates that the custom checkpoint can be trained, loaded, and used for generation through vLLM. It does not yet establish that DFlash improves inference performance.

Priority order:

1. Benchmark target-only Qwen3-8B using the same request as the DFlash run.
2. Repeat both measurements and compare median throughput and latency.
3. Capture actual speculative acceptance statistics.
4. Run the token-count sweep.
5. Implement and validate the runtime kill switch.
6. Evaluate goodput under explicit SLOs.
7. Publish the checkpoint and reproducible benchmark results.

The project's success criterion is **measured, reproducible improvement over the target-only baseline**, with the workload, hardware, trade-offs, and limitations documented.
