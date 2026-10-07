# dflash-qwen3-8b
Custom DFlash speculative decoding project for Qwen3-8B.

This repository focuses on:

- training a custom DFlash speculator
- publishing the trained checkpoint to Hugging Face
- serving Qwen3-8B with DFlash through vLLM
- measuring speculative acceptance
- measuring TTFT, TPOT and throughput
- evaluating goodput under SLOs
- tuning speculative token count
- providing a runtime kill switch
- documenting reproducible experiments

## Architecture

```text
Training data
     |
     v
Qwen3-8B
     |
     +--> hidden states
     |
     +--> target behavior
     |
     v
DFlash training
     |
     v
DFlash checkpoint
     |
     v
Hugging Face
     |
     v
vLLM
     |
     +----------------+
     |                |
     v                v
   DFlash          Qwen3-8B
   draft           verifier
     |                ^
     +------->--------+
             |
             v
       accepted tokens
             |
             v
       runtime metrics
             |
       +-----+------+
       |            |
       v            v
   goodput      kill switch

Locked v0.1 recipe
Parameter	Value
Target model	Qwen/Qwen3-8B
Speculator	DFlash
Auxiliary layers	2, 10, 18, 26, 34
DFlash layers	5
Draft vocabulary	32000
Block size	8
Speculative tokens	7
Max anchors	3072
Sequence length	8192
Epochs	3
Learning rate	6e-4
Scheduler	cosine
Seed	42


The checkpoint revision will be pinned after the trained model is published.
Repository status
Stage 1: repository contract
- [x] Project structure
- [x] Locked configuration
- [x] Dependency pin location
- [x] Result schema
- [x] CPU validation
- [ ] Training
- [ ] Hugging Face checkpoint
- [ ] Serving
- [ ] Acceptance benchmark
- [ ] SLO/goodput benchmark
- [ ] Kill-switch benchmark
- [ ] Public benchmark
