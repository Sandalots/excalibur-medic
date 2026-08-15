#!/usr/bin/env bash

#   ./macintosh/mac_infer.sh  , runs EXCALIBUR-Medic interactively on a Mac, using the llama.cpp server.

# strict, so better to stop than to run something unintended
set -euo pipefail

# the repo root, one level up from macintosh/
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LLAMA="${LLAMA_CPP:-$HOME/llama.cpp}"
MODEL="$ROOT/artifacts/excalibur-medic-1b-Q4_K_M.gguf"
PORT="${PORT:-8080}"

# output helpers
bold() { printf '\033[1m%s\033[0m\n' "$1"; }
warn() { printf '\033[33m%s\033[0m\n' "$1" >&2; }
die()  { printf '\033[31merror: %s\033[0m\n' "$1" >&2; exit 1; }

# the only flag, and the only way past the staleness check below
STALE_OK=0

for a in "$@"; do
  case "$a" in
    --stale-ok) STALE_OK=1 ;;
    *) die "unknown option: $a  (this script takes only --stale-ok)" ;;
  esac
done

# check if there is a llama server
[ -x "$LLAMA/build/bin/llama-server" ] || die "no llama-server at $LLAMA/build/bin
   git clone https://github.com/ggml-org/llama.cpp $LLAMA
   cmake -S $LLAMA -B $LLAMA/build -DCMAKE_BUILD_TYPE=Release
   cmake --build $LLAMA/build -j 10 --target llama-server
   (or set LLAMA_CPP=/path/to/llama.cpp)"

export LLAMA_CPP="$LLAMA"

# check if the model is present
[ -f "$MODEL" ] || die "no $(basename "$MODEL")
   Build it in excalibur.ipynb -- the fuse, convert, quantize and verify sections
   at the end. There is no separate build script; the notebook is the pipeline."

# the build recorded which adapter it came from; compare against the live one
STAMP="$ROOT/artifacts/.build_stamp"
FUSED_W="$ROOT/artifacts/excalibur-medic-1b-fused/model.safetensors"
LIVE="$ROOT/artifacts/checkpoints/final_adapter/adapters.safetensors"

# two ways to be stale: a different adapter, or weights older than it
problem=""

if [ -f "$LIVE" ] && [ -f "$STAMP" ]; then
  want="$(shasum -a 256 "$LIVE" | cut -d' ' -f1)"
  [ "$want" = "$(cat "$STAMP")" ] || problem="the model was built from a DIFFERENT adapter"

elif [ -f "$LIVE" ]; then
  [ -f "$FUSED_W" ] && [ "$LIVE" -nt "$FUSED_W" ] && problem="the fused weights are older than the adapter"
fi

# refuse by default -- running the pre-training model reads as training failure
if [ -n "$problem" ]; then
  warn ""
  warn "  Stale model: $problem."
  [ -f "$FUSED_W" ] && warn "    fused weights  $(date -r "$FUSED_W" '+%Y-%m-%d %H:%M')"
  [ -f "$LIVE" ]    && warn "    adapter        $(date -r "$LIVE" '+%Y-%m-%d %H:%M')"
  warn ""
  warn "  You would be running the model from BEFORE the last training run, so none of"
  warn "  its behaviour would be what the notebook measured."
  warn ""

  if [ "$STALE_OK" = 0 ]; then
    die "rebuild: re-run the build sections of excalibur.ipynb
   — or pass --stale-ok to run the old model anyway"
  fi

  warn "  --stale-ok given; continuing with the old model."
fi

# check if corpus is present
[ -f "$ROOT/datasets/medquad.csv" ] || die "no datasets/medquad.csv — the corpus is missing"

for p in artifacts/drug_labels.csv artifacts/medlineplus.csv; do
  [ -f "$ROOT/$p" ] || warn "  $p absent — retrieval falls back to MedQuAD only"
done

# check if server port is free
# a server already on this port has its own model loaded, which may not be ours
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  holder="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -Fc 2>/dev/null | sed -n 's/^c//p' | head -1)"
  warn "  port $PORT is already held by '$holder'"

  if [ "$holder" = "llama-server" ]; then
    warn "  that server has its own model loaded, which may not be this one — stopping it"
    pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
    sleep 1

  else
    die "port $PORT is in use by '$holder'. Free it, or run with  PORT=8081 $0"
  fi
fi

# detect number of threads
# performance cores only; the efficiency cores slow generation down
PCORES="$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || echo "")"
[ -n "$PCORES" ] || PCORES=$(( $(sysctl -n hw.ncpu) / 2 ))
THREADS="${THREADS:-$PCORES}"

bold ""
CPU="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo Mac)"

bold "EXCALIBUR-Medic — $CPU, ${THREADS} threads"
printf '  model   %s  (%s MB, built %s)\n' "$(basename "$MODEL")" "$(( $(stat -f%z "$MODEL") / 1048576 ))" \
  "$(date -r "$MODEL" '+%d %b %H:%M')"
printf '  ctx     2048, KV q8_0 — the deployed device settings\n'
bold ""

# same runner and same settings as the Pi, so only speed differs
exec python3 excalibur_inference.py --lite --model "$MODEL" --port "$PORT" --ctx 2048 --kv-type q8_0 \
  --threads "$THREADS" --threads-batch "$THREADS"