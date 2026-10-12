# Workload and SLO Definitions

## Initial benchmark matrix

- Concurrency levels: 1, 4, 8, 16
- Target model: Qwen/Qwen3-8B
- Speculation comparison: disabled versus DFlash enabled
- Speculative tokens: 7
- Initial TTFT threshold: 1000 ms
- Initial TPOT threshold: 100 ms
- Maximum configured generation length: 512 tokens
- Configured maximum model length: 16384 tokens

These are initial experiment settings, not measured performance or
guarantees that every configuration fits the selected hardware.

## Traffic classes

The initial classes are coding, reasoning, question answering,
retrieval-augmented generation (RAG), summarization, translation,
writing, structured output and tool calls.

Before a benchmark is considered reproducible, record the dataset or
prompt-suite revision for each class, prompt count, input-token
distribution, output-token limits, generation settings and random seeds
where applicable.

Do not combine results from different classes without identifying the
workload mix.

## SLO semantics

A request meets the initial latency SLO only when both conditions hold:

- TTFT <= 1000 ms
- TPOT <= 100 ms

Goodput fraction = eligible requests meeting both thresholds /
total eligible requests.

Report the eligible-request denominator and the number of failures,
timeouts, cancellations and incomplete requests separately.

Until an explicit policy is implemented, do not silently exclude failed
or timed-out requests or count them as successful.

TPOT measurement must specify whether it excludes the first output
token and how single-token outputs are handled.

## Comparison rules

- Use the same prompts, target model revision, generation settings and
  request schedule for baseline and speculative runs.
- Record software versions, GPU details, model revisions and configuration
  hashes.
- Warm up before measurement and perform at least three independent
  measured repetitions when practical.
- Report run-level results and variability; do not select only the best run.

## Readiness gates

The Hugging Face speculator model reference must be replaced with the
actual published model identifier and immutable revision before serving.

Validate the 16384-token model-length configuration against the actual
model, runtime and available GPU memory before launching the benchmark.
