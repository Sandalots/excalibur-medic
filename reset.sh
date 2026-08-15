#!/usr/bin/env bash

# Return the EXCALIBUR codebase to a clean pre-run state.
#   ./reset.sh          # delete generated artifacts; the fetched corpus stays

# strict: a half-finished delete is worse than none at all
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# takes no options, reject anything passed, rather than ignore it
for a in "$@"; do
  echo "unknown option: $a"; exit 1
done

# anything git tracks is not ours to delete
PROTECTED="$(git ls-files)"

# match the path itself OR anything beneath it -- git tracks
# artifacts/plots/*.png, never the directory, so an exact test
# would delete the tracked figures
protected() {
  printf '%s\n' "$PROTECTED" | grep -qxF "$1" && return 0
  printf '%s\n' "$PROTECTED" | grep -q "^$1/" && return 0
  return 1
}

# everything the notebook derives; the fetched corpus is not here
TARGETS=("artifacts/synthetic" "artifacts/checkpoints" "artifacts/plots" "artifacts/medquad_split.parquet" "artifacts/corpus_combined.csv" "artifacts/bm25_index_v3.pkl" "artifacts/excalibur-medic-1b-fused" "artifacts/imatrix_calib.txt" "artifacts/ppl_heldout.txt" "artifacts/sft_hist.json" "artifacts"/*.gguf)

printf '\n\033[1m== REMOVED\033[0m\n'

total=0
n=0
seen=""

# skip what is absent, already seen, or tracked; size it before removing
for t in "${TARGETS[@]}"; do
  [ -e "$t" ] || continue
  case " $seen " in *" $t "*) continue ;; esac
  seen="$seen $t"
  protected "$t" && continue
  sz="$(du -sk "$t" 2>/dev/null | cut -f1)"
  rm -rf "$t"
  total=$((total + sz))
  n=$((n + 1))
  printf '  %-44s %6s MB\n' "$t" "$((sz / 1024))"
done

# nothing removed means the tree was already clean
if [ "$n" -eq 0 ]; then
  echo "  nothing to remove — already clean"
  exit 0
fi

printf '\n  %d items, %d MB\n' "$n" "$((total / 1024))"

echo "  Next: open excalibur.ipynb with a (.venv) kernel to train the model and get the model deploy ready."