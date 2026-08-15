#!/usr/bin/env bash

# Deploys the EXCALIBUR-Medic model onto the edge Pi device.
#   ./deploy_to_pi.sh , prompts for the address
#   HOST=pi@raspberrypi.local ./deploy_to_pi.sh , skips the prompt

# strict, so a partial deploy leaves the device serving a mixed payload
set -euo pipefail

if [ -z "${HOST:-}" ]; then
  read -rp "Pi address (user@host): " HOST
  [ -n "$HOST" ] || { echo "no address given"; exit 1; }
fi
DEST="${DEST:-excalibur}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# the payload, in the three places the runner expects to find it
MODEL="artifacts/excalibur-medic-1b-Q4_K_M.gguf"

TOP=(excalibur_inference.py benchmark_pi.sh)
DATA=(datasets/medquad.csv)
ART=(artifacts/drug_labels.csv artifacts/drug_aliases.json artifacts/medlineplus.csv artifacts/topic_aliases.json)

# section header
say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

# one ssh connection, reused, so the password is asked for once
CTL=""

sshx() { if [ -n "$CTL" ]; then ssh -o ControlPath="$CTL" "$@"; else ssh "$@"; fi; }
scpx() { if [ -n "$CTL" ]; then scp -o ControlPath="$CTL" "$@"; else scp "$@"; fi; }

say "PRE-FLIGHT"
missing=0

# check the whole payload before opening a connection
for f in "${TOP[@]}" "${DATA[@]}" "${ART[@]}" "$MODEL"; do
  if [ -f "$ROOT/$f" ]; then
    printf '  ok      %-42s %s\n' "$f" "$(du -h "$ROOT/$f" | cut -f1)"

  else
    printf '  MISSING %s\n' "$f"
    missing=1
  fi
done

if [ "$missing" = 1 ]; then
  echo
  echo "  Rebuild what is missing before deploying:"
  echo "    artifacts/drug_labels.csv, drug_aliases.json  -> scripts/fetcher.py drugs"
  echo "    artifacts/medlineplus.csv, topic_aliases.json -> scripts/fetcher.py medlineplus"
  echo "    the GGUF                                      -> see excalibur.ipynb"
  exit 1
fi

say "CONNECTION"
# a control socket, torn down on exit whatever happens
CTL="$(mktemp -u "${TMPDIR:-/tmp}/excalibur-ssh-XXXXXX")"
trap 'ssh -o ControlPath="$CTL" -O exit "$HOST" 2>/dev/null || true; rm -f "$CTL"' EXIT

echo "  connecting to $HOST — enter the password if prompted"

# fail with the three things that are usually wrong, not a stack trace
if ! ssh -o ControlMaster=yes -o ControlPath="$CTL" -o ControlPersist=300 -o ConnectTimeout=15 "$HOST" true; then
  echo
  echo "  could not connect to $HOST."
  echo "    · is the Pi powered on and on the network?   ping ${HOST#*@}"
  echo "    · right user and address?                    HOST=user@address ./deploy_to_pi.sh"
  echo "    · to stop being asked for a password at all:  ssh-copy-id $HOST"
  exit 1
fi

echo "  connected — reusing this connection for every transfer"
sshx "$HOST" "mkdir -p ~/$DEST/artifacts ~/$DEST/datasets"

say "RUNNER AND CORPORA"
for f in "${TOP[@]}"; do
  printf '  %s\n' "$f"
  scpx -q "$ROOT/$f" "$HOST:~/$DEST/"
done

for f in "${DATA[@]}"; do
  printf '  %s\n' "$f"
  scpx -q "$ROOT/$f" "$HOST:~/$DEST/datasets/"
done

for f in "${ART[@]}"; do
  printf '  %s\n' "$f"
  scpx -q "$ROOT/$f" "$HOST:~/$DEST/artifacts/"
done

say "EXCALIBUR-Medic MODEL (~771 MB)"
# 771 MB, so compare checksums and skip the copy when they match
local_sum="$(shasum -a 256 "$ROOT/$MODEL" | cut -d' ' -f1)"
remote_sum="$(sshx "$HOST" "sha256sum ~/$DEST/$MODEL 2>/dev/null | cut -d' ' -f1" || true)"

if [ "$remote_sum" = "$local_sum" ]; then
  echo "  unchanged (${local_sum:0:16}…) — not copied"

else
  [ -n "$remote_sum" ] && echo "  remote differs (${remote_sum:0:16}…) — replacing" || echo "  not present remotely — copying"
                       
  scpx "$ROOT/$MODEL" "$HOST:~/$DEST/artifacts/"
  # verify after copying: a truncated model still starts, then answers badly
  check="$(sshx "$HOST" "sha256sum ~/$DEST/$MODEL | cut -d' ' -f1")"

  if [ "$check" != "$local_sum" ]; then
    echo "  TRANSFER CORRUPT: $check != $local_sum"; exit 1
  fi
  
  echo "  verified ${check:0:16}…"
fi

say "REMOTE STATE"
sshx "$HOST" "cd ~/$DEST && ls -la excalibur_inference.py benchmark_pi.sh \
  datasets/ artifacts/ | sed 's/^/  /'"

# drop the derived corpus and index so the device rebuilds them from
# what was just sent, rather than serving a stale pair
sshx "$HOST" "rm -f ~/$DEST/artifacts/bm25_index_v3.pkl ~/$DEST/artifacts/corpus_combined.csv"
echo "  cleared the derived corpus and index (rebuilt on first run)"

say "NEXT"
cat <<EOF
  ssh $HOST
  cd ~/$DEST && python3 excalibur_inference.py    # interactive; --stats for telemetry
  cd ~/$DEST && ./benchmark_pi.sh                   # full benchmark
  cd ~/$DEST && ./benchmark_pi.sh --energy      # ~50 min: energy + thermal soak
EOF