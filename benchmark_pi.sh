#!/usr/bin/env bash

# EXCALIBUR-Medic — Raspberry Pi 5 benchmarks
#   ./benchmark_pi.sh            throughput: threads, peak memory, BM25 on ARM, end-to-end
#   ./benchmark_pi.sh --energy   energy: idle floor, joules per query, 40-minute thermal soak

# default to the throughput run; --energy switches modes
ENERGY=0

# the one flag this script takes
for a in "$@"; do
  case "$a" in
    --energy) ENERGY=1 ;;
    *) echo "unknown option: $a"; exit 1 ;;
  esac
done

# unset vars are fatal; a failing pipe stage is not, so one dead probe
# does not abandon the whole run
set -uo pipefail

# resolve every path against this script, not the caller's cwd
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-$ROOT/artifacts/excalibur-medic-1b-Q4_K_M.gguf}"
CSV="${CSV:-}"
RAG="${RAG:-$ROOT/excalibur_inference.py}"
BIN="${LLAMA_CPP:-$HOME/llama.cpp}/build/bin"
Q="What are the symptoms of hypothyroidism?"

# tunables, all overridable from the environment
PORT="${PORT:-8080}"
HZ="${HZ:-1}"
DUTY="${DUTY:-20}"
IDLE_BARE_S="${IDLE_BARE_S:-120}"
IDLE_SRV_S="${IDLE_SRV_S:-180}"
NQ="${NQ:-20}"
SUSTAIN_MIN=40

if [ "$ENERGY" = 1 ]; then
  # energy writes a directory of csvs, throughput a single log
  OUT="$ROOT/energy_$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"
  SAMPLES="$OUT/samples.csv"; MARKS="$OUT/marks.csv"; LOG="$OUT/report.txt"

else
  LOG="$ROOT/benchmark_pi_$(date +%Y%m%d-%H%M%S).txt"
fi

# everything below goes to the terminal and the log at once
exec > >(tee "$LOG") 2>&1

# small output helpers
hr() { printf '%s\n' "-------------------------------------------------------------"; }
say() { printf '\n== %s ==\n' "$1"; }
jnum() { printf '%s' "$1" | grep -oE "\"$2\":[0-9.]+" | grep -oE '[0-9.]+$'; }

# poll /health until the server answers or its process dies
wait_ready() {
  local pid="$1" port="$2" secs="$3"

  for _ in $(seq 1 "$secs"); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && return 0
    kill -0 "$pid" 2>/dev/null || return 1
    sleep 1
  done

  return 1
}

note() { printf '   %s\n' "$1"; }
die()  { printf '\033[31merror: %s\033[0m\n' "$1" >&2; exit 1; }

# power sampling 
read_power() {
  vcgencmd pmic_read_adc 2>/dev/null | awk '
    /current\(/ { n=$1; sub(/_A$/,"",n); split($0,p,"="); v=p[2]; sub(/A[ \t]*$/,"",v); amp[n]=v+0 }
    /volt\(/    { n=$1; sub(/_V$/,"",n); split($0,p,"="); v=p[2]; sub(/V[ \t]*$/,"",v); vlt[n]=v+0 }
    END { w=0; for (r in amp) if (r in vlt) w += amp[r]*vlt[r]; printf "%.4f", w }'
}

# timestamp a phase boundary so the report can slice power over it
mark() { printf '%s,%s\n' "$(date +%s.%N)" "$1" >> "$MARKS"; }
jnum() { printf '%s' "$1" | grep -oE "\"$2\":[0-9.]+" | grep -oE '[0-9.]+$'; }

# Sustained load question
SUSTAIN_Q="Explain the pathophysiology of type 2 diabetes."

# background sampler: power, temperature, clock, throttle flags
sampler() {
  local interval; interval=$(awk -v h="$HZ" 'BEGIN{printf "%.3f", 1/h}')
  
  while :; do
    printf '%s,%s,%s,%s,%s\n' "$(date +%s.%N)" "$(read_power)" "$(vcgencmd measure_temp 2>/dev/null | tr -dc '0-9.')" \
      "$(vcgencmd measure_clock arm 2>/dev/null | cut -d= -f2)" \
      "$(vcgencmd get_throttled 2>/dev/null | cut -d= -f2)" >> "$SAMPLES"
    sleep "$interval"
  done
}

# Pre flight checks
# stop the server and the sampler however this exits
SAMPLER_PID=""; SRV_PID=""
cleanup() {
  [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null
  [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null

  if [ "$ENERGY" = 1 ]; then
    printf '\n   samples: %s\n   report : %s\n' "${SAMPLES#$ROOT/}" "${LOG#$ROOT/}"
  
  else
    printf '\noutput saved to: %s\n' "$LOG"
  fi
}

trap cleanup EXIT
if [ "$ENERGY" = 1 ]; then

say "PRE-FLIGHT"
# only a Pi exposes the PMIC, so refuse anything else
command -v vcgencmd >/dev/null 2>&1 || die "no vcgencmd — this script only runs on a Pi"
[ -f "$MODEL" ] || die "no model at $MODEL"
[ -x "$BIN/llama-server" ] || die "no llama-server at $BIN"
[ -f "$RAG" ] || die "no excalibur_inference.py at $RAG"

# read the rails once before committing to a 50-minute run
probe="$(read_power)"

case "$probe" in ""|0|0.0000)
  die "pmic_read_adc returned no usable power.
   check by hand:  vcgencmd pmic_read_adc
   if it prints nothing, this firmware or board does not expose the PMIC ADC and there is
   no software power source available — an inline meter is the only route." ;;
esac

# a plausible idle Pi 5 sits between 0.8 and 25 W
awk -v p="$probe" 'BEGIN{exit !(p>0.8 && p<25)}' || die "pmic_read_adc gave ${probe} W, which is not plausible for a Pi 5. Inspect:  vcgencmd pmic_read_adc"

note "PMIC reads ${probe} W at rest — plausible, proceeding"
note "$(tr -d '\0' < /sys/firmware/devicetree/base/model 2>/dev/null || uname -m)"
note "$(free -m | awk '/^Mem:/{printf "RAM %d MB total, %d MB available", $2, $7}')"
note "sampling at ${HZ} Hz -> ${SAMPLES#$ROOT/}"

# a stray server would distort every phase below
pkill -f "llama-server" 2>/dev/null && { note "stopped a stray llama-server"; sleep 2; }

# start sampling before the first phase begins
sampler & SAMPLER_PID=$!
SRV_PID=""

cleanup() {
  [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null
  kill "$SAMPLER_PID" 2>/dev/null
  printf '\n   samples: %s\n   report : %s\n' "${SAMPLES#$ROOT/}" "${LOG#$ROOT/}"
}

trap cleanup EXIT
sleep 2

# board idle, no server
say "A. BOARD IDLE — no server (${IDLE_BARE_S}s)"
note "the floor everything else is measured against"
mark idle_bare_start; sleep "$IDLE_BARE_S"; mark idle_bare_end

# server resident, idle
say "B. SERVER RESIDENT, IDLE (${IDLE_SRV_S}s)"

note "where an always-on reference device actually spends its life"
# start the server the runner will reuse rather than replace
"$BIN/llama-server" -m "$MODEL" -c 2048 -t 2 -tb 4 -b 128 --jinja -ctk q8_0 -ctv q8_0 --port "$PORT" --host 127.0.0.1 > "$OUT/server.log" 2>&1 &

SRV_PID=$!
ready=0

for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ready=1; break; }
  kill -0 "$SRV_PID" 2>/dev/null || break
  sleep 1
done

[ "$ready" = 1 ] || { tail -6 "$OUT/server.log"; die "llama-server did not start"; }
note "server up on :$PORT (excalibur_inference.py will reuse it rather than start a second)"
sleep 5
mark idle_srv_start; sleep "$IDLE_SRV_S"; mark idle_srv_end

# per query energy
say "C. PER-QUERY ENERGY — ${NQ} questions through retrieval + generation"
note "excalibur_inference.py end to end: BM25, the guards, the model. Not generation alone."

# a fixed question set, so separate runs are comparable
QUESTIONS=(
  "What are the symptoms of gout?" "What causes glaucoma?"
  "How do you treat hypothyroidism?" "Is Marfan syndrome inherited?"
  "What is the dose of metformin?" "What are the symptoms of Lupus?"
  "How is diabetes diagnosed?" "What causes migraines?"
  "What are the side effects of atorvastatin?" "How do you prevent falls?"
  "What is (are) Glaucoma ?" "What causes anemia?"
  "What are the symptoms of asthma?" "How do you treat eczema?"
  "Is cystic fibrosis inherited?" "What causes kidney stones?"
  "What are the warnings for ibuprofen?" "How is anemia treated?"
  "What causes high blood pressure?" "What are the symptoms of shingles?"
)

# the runner resolves its corpus relative to itself
cd "$(dirname "$RAG")" || die "cannot cd to the runner"

# mark each query so the report can integrate power across it
for i in $(seq 0 $((NQ - 1))); do
  q="${QUESTIONS[$((i % ${#QUESTIONS[@]}))]}"
  mark "q${i}_start"
  python3 "$RAG" -q "$q" --quiet --port "$PORT" >/dev/null 2>&1
  mark "q${i}_end"
  printf '   %2d/%d  %s\n' "$((i+1))" "$NQ" "${q:0:52}"
  sleep 3
done

# sustained load
say "D. SUSTAINED LOAD (${SUSTAIN_MIN} min) — does it throttle?"
note "back-to-back generation. The point is not the average watt figure, it is whether"
note "tok/s decays and at what temperature — which single short requests cannot show."
mark sustain_start

# back-to-back generation until the clock runs out
END=$(( $(date +%s) + SUSTAIN_MIN * 60 ))
n=0
: > "$OUT/sustain_rate.csv"

while [ "$(date +%s)" -lt "$END" ]; do
  resp=$(curl -sf -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$SUSTAIN_Q\"}],\"max_tokens\":200,\"temperature\":0}")

  # record tok/s, temperature and throttle state per generation
  rate=$(jnum "$resp" predicted_per_second)
  [ -n "$rate" ] && printf '%s,%s,%s,%s\n' "$(date +%s)" "$rate" "$(vcgencmd measure_temp | tr -dc '0-9.')" \
      "$(vcgencmd get_throttled | cut -d= -f2)" >> "$OUT/sustain_rate.csv"

  n=$((n + 1))
  [ $((n % 10)) -eq 0 ] && printf '   %d generations, %s, %s min left\n' "$n" "$(vcgencmd measure_temp)" "$(( (END - $(date +%s)) / 60 ))"
done

mark sustain_end
note "$n generations"

# stop generating before the report reads the samples
kill "$SRV_PID" 2>/dev/null; SRV_PID=""
sleep 2
kill "$SAMPLER_PID" 2>/dev/null

# produce the report
say "RESULTS"
python3 - "$SAMPLES" "$MARKS" "$OUT/sustain_rate.csv" "$DUTY" <<'PYEOF'
import sys, csv
from pathlib import Path

samples, marks, rates, duty = sys.argv[1:5]
duty = float(duty)

S = []
for r in csv.reader(open(samples)):
    if len(r) < 5 or not r[1]:
        continue
    try:
        S.append((float(r[0]), float(r[1]), float(r[2] or 0), r[4]))
    except ValueError:
        continue
S.sort()
M = {}
for r in csv.reader(open(marks)):
    M.setdefault(r[1], float(r[0]))

if len(S) < 10:
    print("   too few power samples to report on"); raise SystemExit(1)

def window(t0, t1, trim=0.0):
    w = [s for s in S if t0 + trim <= s[0] <= t1 - trim]
    if len(w) < 2:
        return None
    j = sum((w[i+1][0]-w[i][0]) * (w[i+1][1]+w[i][1]) / 2 for i in range(len(w)-1))
    span = w[-1][0] - w[0][0]
    return j, (j/span if span else 0), max(x[1] for x in w), max(x[2] for x in w), len(w)

def show(label, key0, key1):
    if key0 not in M or key1 not in M:
        return None
    r = window(M[key0], M[key1], trim=2.0)
    if not r:
        return None
    j, mean, pk, temp, n = r
    print(f"   {label:<26} {mean:6.2f} W mean   {pk:6.2f} W peak   "
          f"{temp:5.1f} °C peak   ({n} samples)")
    return mean

print("\n  POWER BY PHASE")
bare = show("board idle, no server", "idle_bare_start", "idle_bare_end")
srv = show("server resident, idle", "idle_srv_start", "idle_srv_end")
sus = show("sustained generation", "sustain_start", "sustain_end")

if bare is not None and srv is not None:
    print(f"\n   holding the model resident costs {srv - bare:+.2f} W over a bare board")
if srv is not None and sus is not None:
    print(f"   generating costs {sus - srv:+.2f} W over an idle server")

qs = sorted(int(k[1:-6]) for k in M if k.startswith("q") and k.endswith("_start"))
rows = []
for i in qs:
    a, b = M.get(f"q{i}_start"), M.get(f"q{i}_end")
    if a is None or b is None:
        continue
    r = window(a, b)
    if r:
        rows.append((b - a, r[0]))

guarded = [(b - a) for i in qs
           if (a := M.get(f"q{i}_start")) and (b := M.get(f"q{i}_end")) and (b - a) < 2.0]
if guarded:
    print(f"\n  {len(guarded)} of {len(qs)} questions returned in "
          f"{min(guarded):.2f}-{max(guarded):.2f} s WITHOUT calling the model —")
    print("   refusal, disambiguation or a verbatim quote. Those paths cost no measurable")
    print("   energy; the per-query figures below cover only the ones that generated.")

if rows:
    rows.sort(key=lambda x: x[1])
    med_s, med_j = rows[len(rows)//2]
    idle_ref = srv if srv is not None else (bare or 0)
    marg = [(d, j - idle_ref * d) for d, j in rows]
    marg.sort(key=lambda x: x[1])
    mm = marg[len(marg)//2][1]
    print(f"\n  PER QUERY  (n={len(rows)}, full path: retrieval + guards + generation)")
    print(f"   median latency            {med_s:6.1f} s")
    print(f"   median energy, gross      {med_j:6.1f} J")
    print(f"   median energy, MARGINAL   {mm:6.1f} J   <- the cost of answering")
    print(f"   range, marginal           {marg[0][1]:.1f} - {marg[-1][1]:.1f} J")

p = Path(rates)
if p.exists() and p.stat().st_size:
    R = [(float(a), float(b), float(c), d) for a, b, c, d in csv.reader(open(rates))]
    if len(R) >= 6:
        k = max(3, len(R)//5)
        first, last = R[:k], R[-k:]
        f = sum(x[1] for x in first)/len(first)
        l = sum(x[1] for x in last)/len(last)
        print(f"\n  SUSTAINED LOAD  ({len(R)} generations over "
              f"{(R[-1][0]-R[0][0])/60:.0f} min)")
        print(f"   first {k:>3} generations     {f:6.2f} tok/s   {first[0][2]:.1f} °C")
        print(f"   last  {k:>3} generations     {l:6.2f} tok/s   {last[-1][2]:.1f} °C")
        drop = 100*(f-l)/f if f else 0
        print(f"   change                    {l-f:+6.2f} tok/s ({-drop:+.1f}%)")
        now = [x[3] for x in R]
        hit = [t for t in now if t not in ("0x0", "0x00000000", "")]
        if hit:
            print(f"   get_throttled             {sorted(set(hit))}  <- THROTTLED")
        else:
            print("   get_throttled             clean throughout")
        if abs(drop) < 3 and not hit:
            print("   -> holds its throughput. No thermal ceiling at this duty cycle.")
        elif hit or drop >= 3:
            print("   -> decays under sustained load. Quote the SUSTAINED rate, not the")
            print("      single-request one, for anything longer than a few questions.")

if rows and srv is not None:
    per_h = duty * mm
    avg_w = srv + per_h/3600
    print(f"\n  AT {duty:.0f} QUERIES/HOUR")
    print(f"   average draw              {avg_w:6.2f} W  (idle {srv:.2f} + queries "
          f"{per_h/3600:.2f})")
    print("\n   PMIC-derived and therefore a LOWER BOUND: it excludes PSU conversion loss")
    print("   and anything drawing downstream of the PMIC. Real draw at the wall is higher.")
PYEOF

say "DONE"
echo "   Everything is in ${OUT#$ROOT/}."

exit 0
fi

# throughput mode starts here
say "HOST"
uname -srm
grep -m1 "^Model" /proc/cpuinfo 2>/dev/null || true
nproc | sed 's/^/cores: /'
free -m | awk '/^Mem:/{printf "RAM: %d MB total, %d MB available\n",$2,$7}'

vcgencmd measure_temp 2>/dev/null || true
DT=/sys/firmware/devicetree/base/model
[ -f "$DT" ] && tr -d '\0' < "$DT" && echo

# name everything the benchmark needs rather than assuming it
say "PREREQUISITES"
for f in "$MODEL" "$ROOT/datasets/medquad.csv" "$RAG"; do
  [ -f "$f" ] && printf 'ok      %s (%s)\n' "$f" "$(du -h "$f" | cut -f1)" || printf 'MISSING %s\n' "$f"
done

for b in llama-cli llama-server; do
  [ -x "$BIN/$b" ] && printf 'ok      %s\n' "$BIN/$b" || printf 'MISSING %s/%s\n' "$BIN" "$b"
done

for f in "$ROOT/artifacts/drug_labels.csv" "$ROOT/artifacts/drug_aliases.json"; do
  [ -f "$f" ] && printf 'ok      %s (%s)\n' "$f" "$(du -h "$f" | cut -f1)" || printf 'MISSING %s  <- no drug coverage; scp it from the repo\n' "$f"
done

[ -x "$BIN/llama-cli" ] || { echo "build llama.cpp first (see README)"; exit 1; }

# a running server would distort every row below
if pgrep -f "llama-server" >/dev/null 2>&1; then
  echo "note: a llama-server is already running and would distort every row below."
  pkill -f "llama-server" 2>/dev/null || true
  sleep 2
  echo "      stopped it."
fi

[ -f "$MODEL" ] || { echo; echo "ABORT: model not found at $MODEL"; echo "       pass one with:  MODEL=/path/to/model.gguf ./benchmark_pi.sh"; exit 1; }

# generation speed vs thread count 
say "0. CORPUS THE RUNNER WILL ACTUALLY USE"
python3 - "$ROOT" <<'PYEOF'
import sys, pathlib
sys.path.insert(0, sys.argv[1])

try:
    import excalibur_inference as M
    print(f"  {M.DEFAULT_CSV.name}   aliases: {len(M.ALIASES)}")
    if not M.ALIASES:
        print("  WARNING: no brand-name map -- 'What is Tylenol?' will miss")

except Exception as e:
    print(f"  could not import excalibur_inference: {e}")
PYEOF

# ask the runner which corpus it loads rather than guessing
if [ -z "$CSV" ]; then
  CSV="$(python3 -c "
import sys; sys.path.insert(0, '$ROOT')
try:
    import excalibur_inference as M; print(M.DEFAULT_CSV)

except Exception:
    print('$ROOT/datasets/medquad.csv')" 2>/dev/null)"
fi

[ -f "$CSV" ] || CSV="$ROOT/datasets/medquad.csv"; echo "  benchmarks below use: ${CSV#$ROOT/}"

say "1. GENERATION SPEED vs THREADS (ctx 2048 + q8_0 KV = the deployed config)"
printf '%-9s %-14s %-14s %-12s\n' threads gen_tok/s prompt_tok/s total_s

# sweep thread counts: generation and prompt want different ones
for t in 1 2 3 4; do
  slog="$ROOT/.bench_server_t$t.log"
  "$BIN/llama-server" -m "$MODEL" -c 2048 -t "$t" -b 128 --jinja -ctk q8_0 -ctv q8_0 --port 8098 --host 127.0.0.1 --no-warmup >"$slog" 2>&1 &
  pid=$!

  if wait_ready "$pid" 8098 120; then
    resp=$(curl -sf -m 300 http://127.0.0.1:8098/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"max_tokens\":128,\"temperature\":0}")

    # tok/s from the server's own counters, not from counting frames
    gen=$(jnum "$resp" predicted_per_second | awk '{printf "%.1f", $1}')
    prm=$(jnum "$resp" prompt_per_second    | awk '{printf "%.1f", $1}')
    pms=$(jnum "$resp" predicted_ms)
    tot=$(awk -v m="${pms:-0}" 'BEGIN{printf "%.1f", m/1000}')
    printf '%-9s %-14s %-14s %-12s\n' "$t" "${gen:-?}" "${prm:-?}" "${tot:-?}"
  
  else
    printf '%-9s %-14s %-14s %-12s\n' "$t" "server-failed" "-" "-"
    echo "  --- why (last 6 lines of llama-server output) ---"
    tail -6 "$slog" 2>/dev/null | sed 's/^/  /'
    echo "  ---"
  fi

  # wait for the port to free before the next configuration
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null

  for _ in $(seq 1 15); do
    curl -sf http://127.0.0.1:8098/health >/dev/null 2>&1 || break
    sleep 1
  done
done

echo "note: generation is memory-bandwidth bound and peaks at 2 threads; prompt"
echo "      processing is compute-bound and scales to 4. The Pi 5 is a 2 GB board, headless, with ~1.6 GB usable RAM."

# memory: does it fit the 1.6 GB budget? 
say "2. PEAK MEMORY (llama-server, the deployed path)"

# two configurations: the default, and the one actually deployed
for cfg in "1024:f16:default" "2048:q8_0:quantised-KV"; do
  ctx="${cfg%%:*}"; rest="${cfg#*:}"; kv="${rest%%:*}"; label="${rest#*:}"
  extra=""; [ "$kv" != "f16" ] && extra="-ctk $kv -ctv $kv"
  slog="$ROOT/.bench_mem_$ctx-$kv.log"
  "$BIN/llama-server" -m "$MODEL" -c "$ctx" -t 4 -b 128 --jinja $extra --port 8099 --host 127.0.0.1 --no-warmup >"$slog" 2>&1 &
  pid=$!

  if wait_ready "$pid" 8099 90; then
    curl -sf -m 180 http://127.0.0.1:8099/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"max_tokens\":200}" >/dev/null 2>&1
    # kernel high-water mark, so a transient peak cannot be missed
    rss=$(awk '/VmHWM/{print $2}' "/proc/$pid/status" 2>/dev/null)

    avail=$(free -m | awk '/^Mem:/{print $7}')
    
    if [ -n "${rss:-}" ]; then
      printf 'ctx %-5s KV %-5s (%-13s) peak RSS %6s MB   free while up: %s MB\n' "$ctx" "$kv" "$label" "$((rss/1024))" "$avail"
    
    else
      printf 'ctx %-5s KV %-5s (%-13s) RSS UNREADABLE — /proc/%s gone\n' "$ctx" "$kv" "$label" "$pid"
    fi
  
  else
    printf 'ctx %-5s KV %-5s (%-13s) SERVER FAILED TO START\n' "$ctx" "$kv" "$label"
    echo "  --- why (last 6 lines) ---"; tail -6 "$slog" 2>/dev/null | sed 's/^/  /'; echo "  ---"
  fi

  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null

  for _ in $(seq 1 15); do
    curl -sf http://127.0.0.1:8099/health >/dev/null 2>&1 || break
    sleep 1
  done
done

echo "budget note: 2 GB board, headless, ~1.6 GB usable — anything near that will swap"

# retrieval on ARM 
say "3. BM25 INDEX ON ARM (pure python)"

if [ -f "$RAG" ] && [ -f "$CSV" ]; then
  python3 - "$RAG" "$CSV" <<'PY'
import sys, time, resource, importlib.util
spec = importlib.util.spec_from_file_location("mr", sys.argv[1])
mr = importlib.util.module_from_spec(spec); spec.loader.exec_module(mr)
from pathlib import Path
t0 = time.time(); ix = mr.BM25Index.build(Path(sys.argv[2])); build = time.time() - t0
qs = ["What are the symptoms of hypothyroidism?", "What causes glaucoma?", "How to prevent falls?", "Is Marfan syndrome inherited?"]
t0 = time.time()
for _ in range(25):
    for q in qs: ix.search(q, k=1)
ms = (time.time() - t0) / (25 * len(qs)) * 1000
ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
mb = ru / 1024 if sys.platform.startswith("linux") else ru / 1024 / 1024
print(f"docs {len(ix.docs)}  terms {len(ix.idf)}")
print(f"build {build:.1f}s   query {ms:.1f} ms   python peak RSS {mb:.0f} MB")
PY

else
  echo "skipped (excalibur_inference.py or corpus CSV missing)"
fi

# end-to-end
say "4. END-TO-END WITH RETRIEVAL"
if [ -f "$RAG" ] && [ -f "$CSV" ]; then
  cd "$(dirname "$RAG")" || exit 0
  timeout 300 python3 "$RAG" -q "$Q" --stats --no-reasoning 2>&1 | tail -24

else
  echo "skipped"
fi

hr; say "DONE"
echo "Paste the contents of the log file below back. It is plain text, no personal data."