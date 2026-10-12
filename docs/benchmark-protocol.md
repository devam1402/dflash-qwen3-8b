# DFlash Benchmark Protocol

## 1. Purpose

Compare Qwen3-8B target-only generation against Qwen3-8B with
the trained DFlash speculator under matched workloads.

No performance claim is valid until the run has been measured
on real hardware and its provenance recorded.

## 2. Comparison controls

Keep these consistent between baseline and speculative runs:

- Target model and immutable model revision
- Evaluation prompts and dataset revision
- Output-token limit and generation settings
- Hardware type and available GPU memory
- vLLM version and serving configuration
- Request concurrency and arrival pattern
- Warm-up policy and measurement duration

Record any unavoidable differences explicitly.

## 3. Workload matrix

Run each workload at concurrency 1 first, then increase concurrency
only after validating correctness and stability.

Record:
- Prompt and generated-token length distributions
- Request count and concurrency
- Warm-up request count
- Measured request count
- Repetitions and run duration
- Sampling and stopping settings

Do not compare results from materially different workloads as if
they were directly equivalent.

## 4. Required measurements

For each run, record:

- Output throughput in tokens per second
- Time to first token (TTFT)
- Time per output token (TPOT)
- Goodput and its explicit SLO thresholds
- Speculative acceptance metrics when speculation is enabled
- GPU memory usage and relevant utilization data
- Errors, timeouts and incomplete requests

Document whether latency values are means, medians or percentiles.
For percentile reporting, include p50, p95 and p99 when sample size
supports meaningful estimates.

## 5. Repetitions and reporting

Perform a warm-up before measurement. Run at least three independent
measured repetitions for each comparison condition when practical.
Report individual results and aggregate statistics; do not select
only the best run.

Use identical measurement boundaries for baseline and speculative
serving. Report failed runs and explain exclusions.

## 6. Acceptance metric definitions

Document the exact verifier convention used for tau, including
whether a verifier bonus token is counted.

Keep distributional overlap alpha separate from observed runtime
acceptance. Report per-position acceptance with its denominator.

## 7. Goodput

Define TTFT and TPOT thresholds before running the benchmark.
Goodput is the fraction of eligible requests meeting both thresholds.

Record how errors, timeouts, cancellations and requests with no
generated tokens are handled. Never infer SLO thresholds from results.

## 8. Provenance

Every published run should identify:

- Run type: real or synthetic
- Git commit and configuration SHA-256
- Target model and immutable revision
- Speculator revision
- Dataset identity and revision
- Python, PyTorch, vLLM and Speculators versions
- GPU, GPU memory, driver and CUDA versions
- Workload settings, timestamps and relevant notes

Synthetic fixtures are for software testing only. They are not
performance benchmarks and must never be presented as measured results.

## 9. Publication gate

Publish a speedup or goodput improvement only when:
- Baseline and speculative runs use comparable conditions
- Output correctness has been checked
- Repetitions and variability are disclosed
- The result JSON passes schema validation
- The report identifies the hardware, software and workload
