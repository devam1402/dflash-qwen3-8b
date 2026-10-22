#!/usr/bin/env bash
set -Eeuo pipefail
BASE=/workspace/keen-amber-bison/dflash-qwen3-8b
VENV=/root/dflash-qwen3-8b-venv
UPSTREAM=/root/speculators-upstream
MODEL=Qwen/Qwen3-8B
COMMIT=77871ed62c89499cbbf3c9ad37886e5bd7a9a892
DATA="$BASE/artifacts/data-final"
RESP="$DATA/responses/ultrachat-qwen3-8b.jsonl"
FILTERED="$DATA/responses/ultrachat-qwen3-8b.stop-only.jsonl"
PREP="$DATA/prepared"
HS="$PREP/hidden_states"
TRAIN="$BASE/artifacts/training/dflash-qwen3-8b-full"
LOG="$BASE/logs"
mkdir -p /workspace/keen-amber-bison "$LOG"
nvidia-smi || { echo "GPU/NVML unavailable; stop here."; exit 2; }
avail=$(df -BG /workspace | awk 'NR==2 {gsub("G","",$4); print $4}')
if [ "${avail:-0}" -lt 35 ]; then echo "Need >=35 GB free on persistent /workspace; currently ${avail} GB."; df -h /workspace; exit 2; fi
if [ ! -d "$BASE/.git" ]; then git clone https://github.com/devam1402/dflash-qwen3-8b.git "$BASE"; fi
cd "$BASE"
mkdir -p "$DATA/responses" "$PREP" "$TRAIN" "$LOG"
export HF_HOME=/root/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
if [ ! -x "$VENV/bin/python" ]; then python3 -m venv "$VENV"; fi
"$VENV/bin/python" -m pip install -U pip setuptools wheel
"$VENV/bin/python" -m pip install "vllm==0.31.0" "transformers==5.16.1"
if [ ! -d "$UPSTREAM/.git" ]; then git clone https://github.com/vllm-project/speculators.git "$UPSTREAM"; fi
git -C "$UPSTREAM" fetch --all
git -C "$UPSTREAM" checkout "$COMMIT"
"$VENV/bin/python" -m pip install -e "$UPSTREAM"
"$VENV/bin/python" -m pip install -e "$BASE" pytest
"$VENV/bin/python" - <<'PY'
import torch, transformers, vllm
from importlib.metadata import version
print("torch", torch.__version__, "transformers", transformers.__version__, "vllm", vllm.__version__, "speculators", version("speculators"))
assert torch.cuda.is_available(), "CUDA unavailable"
print("GPU:", torch.cuda.get_device_name(0))
PY
"$VENV/bin/python" - "$UPSTREAM" <<'PY'
import pathlib, re, sys
r=pathlib.Path(sys.argv[1])
p=r/"src/speculators/models/dflash/core.py"; s=p.read_text()
if 'val_kwargs["max_anchors"] = min(max_anchors, 512)' not in s:
    if "return train_kwargs, val_kwargs" not in s: raise SystemExit("Could not patch DFlash validation kwargs")
    s=s.replace("return train_kwargs, val_kwargs",'val_kwargs["max_anchors"] = min(max_anchors, 512)\n        return train_kwargs, val_kwargs',1); p.write_text(s)
p=r/"src/speculators/train/checkpointer.py"; s=p.read_text()
pattern=r"def convert_float_dtype\(sd: pytree\.PyTree, dtype: torch\.dtype\) -> pytree\.PyTree:\n.*?(?=\n\ndef |\Z)"
replacement='''def convert_float_dtype(sd: pytree.PyTree, dtype: torch.dtype) -> pytree.PyTree:
    def convert_fn(x):
        if isinstance(x, torch.Tensor):
            if x.is_floating_point():
                return x.to(device="cpu", dtype=dtype)
            return x.to(device="cpu")
        return x
    return pytree.tree_map(convert_fn, sd)
'''
if 'return x.to(device="cpu", dtype=dtype)' not in s:
    s,n=re.subn(pattern,replacement,s,count=1,flags=re.S)
    if n!=1: raise SystemExit("Could not patch checkpoint CPU conversion")
    p.write_text(s)
print("Validation cap and CPU checkpoint-save patch verified.")
PY
"$VENV/bin/python" -m compileall -q "$UPSTREAM/src/speculators"
SERVER_PID=
stop_server() {
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for i in $(seq 1 20); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 2; done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=
}
trap stop_server EXIT INT TERM
start_server() {
  mode="$1"; logfile="$LOG/vllm-$mode.log"; stop_server
  if [ "$mode" = extract ]; then
    mkdir -p "$HS"
    nohup "$VENV/bin/python" -m vllm.entrypoints.cli.main serve "$MODEL" \
      --speculative_config '{"method":"extract_hidden_states","num_speculative_tokens":1,"draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":[2,10,18,26,34,36]}}}' \
      --kv_transfer_config "{\"kv_connector\":\"ExampleHiddenStatesConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"shared_storage_path\":\"$HS\"}}" \
      --enable-scale-out --api-server-count 1 --renderer-num-workers 2 --port 8000 \
      --gpu-memory-utilization 0.85 --max-model-len 8192 >"$logfile" 2>&1 &
  else
    nohup "$VENV/bin/python" -m vllm.entrypoints.cli.main serve "$MODEL" --host 127.0.0.1 --port 8000 --gpu-memory-utilization 0.85 --max-model-len 8192 --max-num-seqs 4 >"$logfile" 2>&1 &
  fi
  SERVER_PID=$!
  echo "vLLM $mode PID=$SERVER_PID; log=$logfile"
  for i in $(seq 1 180); do
    if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then return 0; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then tail -80 "$logfile"; return 1; fi
    sleep 5
  done
  tail -80 "$logfile"; return 1
}
if [ ! -s "$FILTERED" ]; then
  start_server normal
  "$VENV/bin/speculators" regenerate-responses --endpoint http://127.0.0.1:8000/v1/chat/completions --model "$MODEL" --dataset ultrachat --split train_sft --limit 500 --concurrency 4 --max-tokens 8192 --sampling-params '{"temperature":0.6,"top_p":0.95,"seed":42}' --seed 42 --outfile "$RESP" --resume 2>&1 | tee "$LOG/regenerate-responses.log"
  stop_server
  "$VENV/bin/python" - "$RESP" "$FILTERED" <<'PY'
import json, pathlib, sys
src,dst=map(pathlib.Path,sys.argv[1:]); n=0
with src.open() as fi,dst.open("w") as fo:
 for line in fi:
  try: row=json.loads(line)
  except json.JSONDecodeError: continue
  meta=row.get("metadata") or {}
  sampling=meta.get("sampling_params") or {}
  kwargs=sampling.get("chat_template_kwargs") or {}
  if meta.get("finish_reason")=="stop" and kwargs.get("enable_thinking") is False:
   fo.write(json.dumps(row,ensure_ascii=False)+"\n"); n+=1
print("Filtered stop-only rows:",n)
if n<100: raise SystemExit("Too few valid rows; inspect regeneration log.")
PY
fi
if [ ! -s "$PREP/d2t-32000.npy" ] || [ ! -s "$PREP/token_freq.pt" ]; then
  "$VENV/bin/speculators" prepare-data --model "$MODEL" --data "$FILTERED" --output "$PREP" --seq-length 8192 --seed 42 --num-preprocessing-workers 1 --no-skip-token-freq 2>&1 | tee "$LOG/prepare-data.log"
fi
[ -s "$PREP/d2t-32000.npy" ] && [ -s "$PREP/t2d-32000.npy" ] || { echo "Missing vocab mappings."; exit 3; }
count=$(find "$HS" -maxdepth 1 -type f -name '*.safetensors' 2>/dev/null | wc -l)
if [ "$count" -lt 800 ]; then
  start_server extract
  "$VENV/bin/speculators" generate-offline-data --model "$MODEL" --endpoint http://127.0.0.1:8000/v1 --preprocessed-data "$PREP" --output "$HS" --concurrency 1 --validate-outputs --fail-on-error 2>&1 | tee "$LOG/generate-offline-data.log"
  stop_server
fi
count=$(find "$HS" -maxdepth 1 -type f -name '*.safetensors' 2>/dev/null | wc -l)
echo "Hidden-state files: $count"
[ "$count" -ge 800 ] || { echo "Insufficient hidden states; training not started."; exit 4; }
"$VENV/bin/speculators" train --speculator-type dflash --verifier-name-or-path "$MODEL" --num-layers 5 --draft-arch qwen3 --target-layer-ids 2 10 18 26 34 --draft-vocab-size 32000 --data-path "$PREP" --hidden-states-path "$HS" --on-missing raise --total-seq-len 8192 --max-anchors 3072 --block-size 8 --loss-fn ce --per-position-loss-weight dpace --dpace-alpha 0.5 --optimizer muon --lr 0.0006 --weight-decay 0.01 --muon-lr 0.0006 --muon-momentum 0.95 --muon-weight-decay 0.1 --muon-ns-steps 5 --muon-adjust-lr-fn match_rms_adamw --scheduler-type cosine --epochs 3 --seed 42 --gradient-checkpointing --num-workers 0 --prefetch-factor 1 --save-path "$TRAIN" 2>&1 | tee "$LOG/train-full.log"
echo "Pipeline complete. Checkpoints: $TRAIN"
