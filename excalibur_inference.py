#!/usr/bin/env python3
"""
EXCALIBUR-Medic, inference runtime script for the Pi 5 device.

A 1B model answers only from documents retrieved at query time; five guards decide,
before generation, whether a suitable source was found. It answers what it has indexed
and does not generalise past it, with a train-only index, retrieval measured no better
than none at all, which is the boundary the guards and the refusal path exist for.
"""
from __future__ import annotations
import argparse, csv, difflib, json, math, os, pickle, re, signal, subprocess, sys
import threading, time
import urllib.request, urllib.error
from array import array
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "artifacts" / "excalibur-medic-1b-Q4_K_M.gguf"

_MEDQUAD  = ROOT / "datasets" / "medquad.csv"               # NIH conditions
_DRUGS    = ROOT / "artifacts" / "drug_labels.csv"          # openFDA labels
_MPLUS    = ROOT / "artifacts" / "medlineplus.csv"          # NLM health topics
_PARTS    = (_MEDQUAD, _DRUGS, _MPLUS)
_COMBINED = ROOT / "artifacts" / "corpus_combined.csv"      # derived from the above

def _corpus() -> Path:
    # staleness check: without it a deploy that added medlineplus.csv kept indexing
    # 17,307 docs instead of 19,411, and the stamp reported "cached" at every layer.
    # derive identity from the INPUTS, not from an output existing
    if _COMBINED.exists():
        newest = max((p.stat().st_mtime for p in _PARTS if p.exists()), default=0)

        if _COMBINED.stat().st_mtime >= newest:
            return _COMBINED
        
        print(f"{_COMBINED.name} is older than its sources — rebuilding", file=sys.stderr)

    missing = [p.name for p in _PARTS[1:] if not p.exists()]

    if missing:
        print(f"warning: {', '.join(missing)} missing — reduced coverage. Rebuild with " f"scripts/fetcher.py all", file=sys.stderr)
        
    present = [p for p in _PARTS if p.exists()]

    if len(present) < 2:
        return _MEDQUAD
    
    fields = ["question", "answer", "source", "focus_area"]

    _COMBINED.parent.mkdir(parents=True, exist_ok=True)

    with open(_COMBINED, "w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=fields); w.writeheader()

        for part in present:
            with open(part, newline="", encoding="utf-8") as f:

                for row in csv.DictReader(f):
                    w.writerow({k: (row.get(k) or "") for k in fields})

    print(f"built {_COMBINED.name} from {' + '.join(p.name for p in present)}", file=sys.stderr)

    return _COMBINED

DEFAULT_CSV = _corpus()
INDEX_CACHE = ROOT / "artifacts" / "bm25_index_v3.pkl"   # v3: facet + condition aware
SERVER_BIN = (Path(os.environ.get("LLAMA_CPP") or Path.home() / "llama.cpp") / "build" / "bin" / "llama-server")

SYSTEM_PROMPT = ( "You are EXCALIBUR-Medic, an offline clinical reference assistant. Reason briefly " "inside <reasoning> tags, then give a clear answer inside <answer> tags. When " "reference material is provided, answer only from it. If a question requires " "examination, imaging, or labs you do not have, say so plainly. Never invent drug " "dosages or statistics. You are not a substitute for a clinician." )

STOP = set("""the a an and or of to in for with is are was were be been being on at by from as that this these those it its can may might will would should could have has had not no if you your patient patients treatment symptoms cause causes disease condition what who how does do did there their them then than when where which while about into over under""".split())

_TOK = re.compile(r"[a-z][a-z0-9]{2,}")
# Eponyms as the corpus writes them: "Alzheimer's", "Crohn's". Feeds BM25Index.poss.

_POSS = re.compile(r"\b([a-z]{3,})'s\b")
# A one-word alias in this many documents or more is ordinary vocabulary, not a
# brand. See prune_aliases() for the measurement behind the number.
_ALIAS_MAX_DF = 10

# retrieval uses function words only. the larger STOP list scores answer overlap,
# but in retrieval "symptoms" and "causes" are the primary discriminator between
# entries for the SAME condition. this list alone took top-1 from 49.2% to 55.0%

RETRIEVAL_STOP = set("""the a an and or of to in for with is are was were be been being on at by from as that this these those it its what who how does do did there their them then than when where which while about into over under""".split())

QUESTION_WEIGHT = 6      # MedQuAD question titles carry the condition name
FACET_BOOST = 0.5        # acts as a tie-break; 0.5 and 2.0 measured identically
CONDITION_BOOST = 1.5    # see condition_of() below -- the single biggest retrieval win
BM25_K1, BM25_B = 1.2, 0.4
GENERAL_TOPIC_WEIGHT = 0.90

LITE = False

# `sideeffect` first, ahead of `cause`: "what adverse reactions does X cause"
# otherwise resolves to `cause` and stops being quoted verbatim. zero corpus
# questions match both patterns, so no document changes facet
_FACETS = [("sideeffect", r"side effect|adverse (reaction|effect)"),
           ("symptom", r"symptom|sign"), ("treat", r"treatment|treat\b|therap"),
           ("cause", r"\bcause"), ("genetic", r"genetic|inherit|mutation"),
           ("risk", r"at risk|how many|frequency|prevalen"), ("prevent", r"prevent"),
           ("diagnos", r"diagnos|test for|exams"),
           ("outlook", r"outlook|prognos"), ("research", r"research|clinical trial"),
           ("dose", r"\bdose|dosage|how much .* take|how many mg|how often .* take"),
           ("interact", r"interact|take .* (with|together)|combined with"),
           ("contra", r"who should not|should i avoid|contraindicat"),
           ("warning", r"\bwarning|precaution|when should i stop"),
           ("pregnancy", r"pregnan|breast ?feed|lactation|nursing|safe (in|during|while)"),
           ("define", r"what is|what are|do you have information about")]

# the condition name extracts from 97.7% of entries, and boosting matching documents
# collapses the wrong-condition errors -- 36% of misses. "functional" = same condition
# AND question type, which is 64% of apparent failures under a strict exact-row metric
_COND_PATS = [
    r"what is \(are\) (.+?)\s*\?", r"what are the symptoms of (.+?)\s*\?",
    r"what causes (.+?)\s*\?", r"what are the treatments for (.+?)\s*\?",
    r"how to diagnose (.+?)\s*\?", r"how to prevent (.+?)\s*\?",
    r"who is at risk for (.+?)[\s?]*$", r"is (.+?) inherited\s*\?",
    r"what are the genetic changes related to (.+?)\s*\?",
    r"how many people are affected by (.+?)\s*\?", r"what is the outlook for (.+?)\s*\?",
    r"do you have information about (.+?)\s*$", r"what research .* for (.+?)\s*\?",
    r"how (?:do (?:you|i) |to |can (?:you|i) )?(?:treat|cure|manage) (.+?)\s*\?*$",
    r"what to do for (.+?)\s*\?*$",
    r"what is the dose of (.+?)\s*\?*$", r"what are the side effects of (.+?)\s*\?*$",
    r"what interacts with (.+?)\s*\?*$", r"who should not take (.+?)\s*\?*$",
    r"what are the (?:serious )?warnings for (.+?)\s*\?*$",
    r"what warnings apply to (.+?)\s*\?*$",
    r"what precautions apply to (.+?)\s*\?*$",
    r"when should i stop taking (.+?)\s*\?*$",
    r"is (.+?) safe in pregnancy\s*\?*$", r"is (.+?) safe while breast ?feeding\s*\?*$",
    r"how much (.+?) (?:can|should) i take\s*\?*$",
    r"treatments? (?:for|of) (.+?)\s*\?*$",
    r"what(?:\'s| is| are)(?: \(are\))? (?:the )?(.+?)\s*\?*$",
    r"tell me about (.+?)\s*\?*$",
]

def condition_of(text: str) -> str | None:
    t = text.lower().strip()

    for pat in _COND_PATS:
        m = re.search(pat, t)

        if m:
            c = m.group(1).strip(" ?.")
            # leading articles are deliberately NOT stripped. "what is a ct scan"
            # misses "CT Scans", but stripping measured worse overall (90.1% ->
            # 87.3%): the bare form collides with plural titles instead

            return re.sub(r"\s*\(also known as[^)]*\)", "", c).strip() or None
        
    return None

def facet_of(text: str) -> str:
    t = text.lower()

    for name, pat in _FACETS:

        if re.search(pat, t):
            return name
        
    return "other"

def tokenize(text: str) -> list[str]:
    return [w for w in _TOK.findall(text.lower()) if w not in RETRIEVAL_STOP]

# index
class BM25Index:
    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1, self.b = k1, b
        self.stamp: str | None = None      # corpus identity; see corpus_stamp()
        self.docs: list[tuple[str, str]] = []      # (question, answer)
        self.dfacet: list[str] = []
        self.dcond: list[str] = []
        self.dgeneral: list[bool] = []     # True for broad consumer-health topics
        self.poss: dict[str, str] = {}     # "alzheimers" -> "alzheimer's"; see build()
        self.post_ids: dict[str, array] = {}
        self.post_tf: dict[str, array] = {}
        self.idf: dict[str, float] = {}
        self.doclen = array("f")
        self.avglen = 1.0

    @classmethod
    def build(cls, csv_path: Path) -> "BM25Index":
        ix = cls()
        seen = set()

        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                q, a = (row.get("question") or "").strip(), (row.get("answer") or "").strip()

                if not q or not a or len(a) < 120:
                    continue

                key = (q, a[:200])

                if key in seen:
                    continue

                seen.add(key)
                ix.docs.append((q, a))
                ix.dgeneral.append("MedlinePlus" in (row.get("source") or ""))

        tmp_ids: dict[str, list] = {}
        tmp_tf: dict[str, list] = {}
        ix.dfacet = [facet_of(q) for q, _ in ix.docs]
        ix.dcond = [condition_of(q) or "" for q, _ in ix.docs]
        # an apostrophe is a token boundary, so "Alzheimer's" and "Alzheimers" are
        # unrelated tokens no stemmer reconciles -- 69 of 70 possessive names served
        # with it, 16 without. not a stemmer itself: it learns "xs -> x's" only for
        # words this corpus writes possessively, never for an ordinary plural

        for q, _ in ix.docs:
            for m in _POSS.finditer(q.lower()):
                ix.poss[m.group(1) + "s"] = m.group(1) + "'s"

        for i, (q, a) in enumerate(ix.docs):
            # weight the question field by repeating it: titles carry the condition name
            counts = Counter(tokenize(q) * QUESTION_WEIGHT + tokenize(a))
            ix.doclen.append(float(sum(counts.values())))

            for w, f_ in counts.items():
                tmp_ids.setdefault(w, []).append(i)
                tmp_tf.setdefault(w, []).append(min(f_, 32767))

        n = len(ix.docs)
        ix.avglen = (sum(ix.doclen) / n) if n else 1.0

        for w, ids in tmp_ids.items():
            ix.post_ids[w] = array("i", ids)
            ix.post_tf[w] = array("i", tmp_tf[w])

            df = len(ids)
            ix.idf[w] = math.log(1 + (n - df + 0.5) / (df + 0.5))

        return prune_aliases(ix)

    def _bm25(self, query: str) -> dict[int, float]:
        scores: dict[int, float] = {}

        for w in set(tokenize(query)):
            ids = self.post_ids.get(w)

            if ids is None:
                continue

            idf, tfs = self.idf[w], self.post_tf[w]

            for j, doc in enumerate(ids):
                f_ = tfs[j]
                dl = self.doclen[doc]
                denom = f_ + self.k1 * (1 - self.b + self.b * dl / self.avglen)

                scores[doc] = scores.get(doc, 0.0) + idf * f_ * (self.k1 + 1) / denom

        return scores

    def _boost(self, scores: dict[int, float], query: str) -> None:
        """The three tuned adjustments, in place. Each was measured on held-out questions."""
        # MedlinePlus topics are short and broad, and BM25 length normalisation
        # rewards that -- "primary hyperparathyroidism" answered from "Parathyroid
        # Disorders", functional@1 100.0% -> 96.8%. they fill gaps, not compete
        if scores and self.dgeneral:
            for doc in scores:

                if doc < len(self.dgeneral) and self.dgeneral[doc]:
                    scores[doc] *= GENERAL_TOPIC_WEIGHT

        if scores and self.dfacet:
            qf = facet_of(query)

            for d in scores:
                if self.dfacet[d] == qf:
                    scores[d] *= (1.0 + FACET_BOOST)

        if scores and self.dcond and CONDITION_BOOST:
            qc = condition_of(query)

            if qc:
                for d in scores:
                    dc = self.dcond[d]

                    if not dc:
                        continue

                    if dc == qc:
                        scores[d] *= (1.0 + CONDITION_BOOST)

                    elif dc in qc:
                        # doc is BROADER than the question ("glaucoma" for a query about
                        # "early-onset glaucoma") -- a reasonable fallback.
                        scores[d] *= (1.0 + CONDITION_BOOST * 0.5)

                    elif qc in dc:
                        # doc is NARROWER than the question ("skin cancer" for
                        # "cancer") -- answering from it substitutes a different
                        # question, so barely boost it
                        scores[d] *= (1.0 + CONDITION_BOOST * 0.15)

    def search(self, query: str, k: int = 1) -> list[tuple[float, int]]:
        scores = self._bm25(query)
        self._boost(scores, query)

        return sorted(((s, d) for d, s in scores.items()), reverse=True)[:k]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump({"stamp": self.stamp, "k1": self.k1, "b": self.b, "docs": self.docs, "dfacet": self.dfacet, "dcond": self.dcond, "dgeneral": self.dgeneral, "poss": self.poss, "post_ids": self.post_ids, "post_tf": self.post_tf, "idf": self.idf, "doclen": self.doclen, "avglen": self.avglen}, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> "BM25Index":
        with open(path, "rb") as f:
            d = pickle.load(f)

        if not isinstance(d, dict):
            raise ValueError("legacy cache format")
        
        ix = BM25Index(d["k1"], d["b"])
        for k in ("docs", "dfacet", "dcond", "post_ids", "post_tf", "idf", "doclen", "avglen"):
            setattr(ix, k, d[k])

        ix.dgeneral = d.get("dgeneral") or [False] * len(ix.docs)
        ix.poss = d.get("poss") or {} # absent in a pre-possessive cache
        ix.stamp = d.get("stamp")

        return prune_aliases(ix)

def corpus_stamp(csv_path: Path) -> str:
    try:
        st = csv_path.stat()

        return f"{csv_path.name}:{st.st_size}:{int(st.st_mtime)}"
    
    except OSError:
        return "missing"

def get_index(csv_path: Path, cache: Path, rebuild: bool = False) -> BM25Index:
    want = corpus_stamp(Path(csv_path))

    if cache.exists() and not rebuild:
        try:
            t0 = time.time()
            ix = BM25Index.load(cache)

            if getattr(ix, "stamp", None) != want:
                print(f"index cache is stale (built from {getattr(ix, 'stamp', None)}, " f"corpus is {want}); rebuilding", file=sys.stderr)
                
                raise ValueError("stale")
            
            if not LITE:
                print(f"index: {len(ix.docs)} docs, {len(ix.idf)} terms " f"(cached, {time.time()-t0:.1f}s)", file=sys.stderr)
                
            return ix
        
        except Exception as e:
            if str(e) != "stale":
                print(f"index cache unreadable ({e}); rebuilding", file=sys.stderr)

    t0 = time.time()
    ix = BM25Index.build(csv_path)
    ix.stamp = want
    ix.save(cache)

    if not LITE:
        print(f"index: {len(ix.docs)} docs, {len(ix.idf)} terms " f"(built in {time.time()-t0:.1f}s, cached to {cache.name})", file=sys.stderr)
        
    return ix

class S:
    RESET = DIM = BOLD = ITAL = CYAN = YELLOW = GREEN = RED = BLUE = GREY = ""
    WHITE = MAGENTA = ""

    @classmethod
    def enable(cls, mode: str = "auto") -> None:
        on = (mode == "always") or ( mode == "auto" and sys.stdout.isatty() and os.environ.get("TERM", "") not in ("", "dumb") and "NO_COLOR" not in os.environ )

        if not on:
            return
        
        cls.RESET, cls.DIM, cls.BOLD, cls.ITAL = "\033[0m", "\033[2m", "\033[1m", "\033[3m"
        cls.CYAN, cls.YELLOW, cls.GREEN = "\033[36m", "\033[33m", "\033[32m"
        cls.RED, cls.BLUE, cls.GREY = "\033[31m", "\033[94m", "\033[90m"
        cls.WHITE, cls.MAGENTA = "\033[97m", "\033[95m"

def rule(label: str = "", width: int = 64, colour: str = "") -> str:
    c = colour or S.BLUE

    if not label:
        return f"{c}{'─' * width}{S.RESET}"
    
    return f"{c}── {label} {'─' * max(0, width - len(label) - 4)}{S.RESET}"

class TagStyler:
    OPEN = {"<reasoning>": "reasoning", "<answer>": "answer"}
    CLOSE = {"</reasoning>", "</answer>"}
    _MAXLEN = max(len(t) for t in list(OPEN) + list(CLOSE))

    def __init__(self, out=None):
        self.out = out or sys.stdout
        self.buf, self.section, self.started = "", None, False

    _STYLE = {"reasoning": lambda: S.BLUE, "answer": lambda: S.WHITE + S.BOLD}

    def _header(self, name: str) -> str:
        if name == "reasoning":
            return f"\n{rule('reasoning', colour=S.GREY)}\n"
        
        return f"\n{rule('ANSWER', colour=S.GREEN)}\n"

    def feed(self, chunk: str) -> None:
        self.buf += chunk

        while True:
            hit = None

            for tag in list(self.OPEN) + list(self.CLOSE):
                i = self.buf.find(tag)

                if i != -1 and (hit is None or i < hit[0]):
                    hit = (i, tag)

            if hit is None:
                break

            i, tag = hit

            if i:
                self.out.write(self.buf[:i])

            self.buf = self.buf[i + len(tag):]

            if tag in self.OPEN:
                self.section = self.OPEN[tag]
                self.out.write(self._header(self.section))
                self.out.write(self._STYLE[self.section]())

            else:
                self.out.write(S.RESET)
                self.section = None

        # hold back anything that could still become a tag
        keep = 0
        for n in range(min(self._MAXLEN - 1, len(self.buf)), 0, -1):
            if any(t.startswith(self.buf[-n:]) for t in list(self.OPEN) + list(self.CLOSE)):
                keep = n
                
                break

        if len(self.buf) > keep:
            self.out.write(self.buf[:len(self.buf) - keep])
            self.buf = self.buf[len(self.buf) - keep:]

        self.out.flush()

    def flush(self) -> None:
        if self.buf:
            self.out.write(self.buf)
            self.buf = ""

        self.out.write(S.RESET + "\n")
        self.out.flush()

def startup_report(args, ix) -> None:
    e = sys.stderr
    w = 64

    def row(k, v, colour=""):
        print(f"  {S.CYAN}{k:<13}{S.RESET}{colour}{v}{S.RESET}", file=e)

    print(f"\n{S.BOLD}{S.WHITE}  EXCALIBUR-Medic{S.RESET}", file=e)
    print(f"  {rule('', w)}", file=e)

    if ix is not None:
        topics = len({c for c in ix.dcond if c})
        row("corpus", f"{S.WHITE}{DEFAULT_CSV.name}{S.RESET}  {len(ix.docs):,} docs · "
                      f"{len(ix.idf):,} terms · {topics:,} topics")
    else:
        row("corpus", "retrieval DISABLED (--no-rag)", S.YELLOW)

    row("server", f"ctx {args.ctx} · KV {args.kv_type} · {args.threads} gen / "
                  f"{args.threads_batch} prompt threads · temp {args.temp}")
    row("guards", f"score ≥ {args.min_score} · coverage ≥ {args.min_coverage} · "
                  f"topic ≥ {args.min_topic}")

    # host — the numbers that decide whether a timing is trustworthy
    bits = []

    if os.cpu_count():
        bits.append(f"{os.cpu_count()} cores")

    avail = mem_available_mb()

    if avail:
        bits.append(f"{avail:,.0f} MB free")

    mhz = cpu_mhz()

    if mhz:
        bits.append(f"{mhz:.0f} MHz")

    t = soc_temp_c()

    if t:
        bits.append(f"{t:.1f} °C")

    if bits:
        row("host", " · ".join(bits))

    print(f"  {rule('', w)}", file=e)

    print("  A 1B model distilled from an 8B medical teacher, making heavy use of RAG", file=e)
    print("  (retrieval-augmented generation): every prompt is looked up in an indexed", file=e)
    print("  medical corpus first, and the answer written from what comes back.", file=e)

    print(f"\n  {S.BOLD}{S.WHITE}Ask about a named condition or drug{S.RESET} — the more "
          f"specific, the better.", file=e)
    
    for ex in ("What are the symptoms of hypothyroidism?",
               "What causes anemia?",
               "How do you treat gout?",
               "What is PTSD?",
               "What are the symptoms of Crohn's disease?",
               "What is an MRI?",
               "What is the dose of metformin?",
               "Who should not take warfarin?",
               "Is lisinopril safe in pregnancy?",
               "What is Lipitor?"):
        
        print(f"    {S.CYAN}▸ {ex}{S.RESET}", file=e)

    print(f"\n  If nothing relevant is indexed it says so rather than guessing, and offers", file=e)
    print("  more specific topics when your question is too broad.", file=e)

    print(f"\n  {rule('', w)}", file=e)
    print(f"  {S.GREY}Ctrl-C or 'exit' to quit · '?' for commands · 'stats' for "
          f"timings{S.RESET}", file=e)

_CLK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100

def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read()
        
    except OSError:
        return None

def _vcgencmd(*args: str) -> str | None:
    try:
        out = subprocess.run(["vcgencmd", *args], capture_output=True, text=True, timeout=2)

        return out.stdout if out.returncode == 0 else None
    
    except Exception:
        return None

def board_power_w() -> float | None:
    txt = _vcgencmd("pmic_read_adc")

    if not txt:
        return None
    
    amps: dict[str, float] = {}
    volts: dict[str, float] = {}

    for line in txt.splitlines():
        m = re.search(r"(\S+?)_([AV])\s+\w+\(\d+\)=([0-9.]+)([AV])", line.strip())

        if not m:
            continue

        rail, kind, val = m.group(1), m.group(2), float(m.group(3))
        (amps if kind == "A" else volts)[rail] = val

    if not amps or not volts:
        return None
    
    return sum(v * amps[r] for r, v in volts.items() if r in amps) or None

def cpu_jiffies() -> tuple[int, int] | None:
    txt = _read("/proc/stat")

    if not txt:
        return None
    
    parts = txt.split("\n", 1)[0].split()[1:]

    try:
        vals = [int(x) for x in parts]

    except ValueError:
        return None
    
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0) # idle + iowait

    return sum(vals) - idle, sum(vals)

def proc_cpu(pid: int) -> float | None:
    txt = _read(f"/proc/{pid}/stat")

    if not txt:
        return None
    
    try:
        fields = txt[txt.rindex(")") + 2:].split()

        return (int(fields[11]) + int(fields[12])) / _CLK # 14th/15th overall
    
    except (ValueError, IndexError):
        return None

def proc_mem_mb(pid: int) -> tuple[float | None, float | None]:
    """(current RSS, peak RSS) in MB for a pid."""
    txt = _read(f"/proc/{pid}/status")

    if not txt:
        return None, None
    
    cur = peak = None

    for line in txt.splitlines():
        if line.startswith("VmRSS:"):
            cur = int(line.split()[1]) / 1024

        elif line.startswith("VmHWM:"):
            peak = int(line.split()[1]) / 1024

    return cur, peak

def mem_available_mb() -> float | None:
    txt = _read("/proc/meminfo")

    if not txt:
        return None
    
    for line in txt.splitlines():

        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
        
    return None

def soc_temp_c() -> float | None:
    raw = _read("/sys/class/thermal/thermal_zone0/temp")

    if raw:

        try:
            return int(raw.strip()) / 1000
        
        except ValueError:
            pass

    txt = _vcgencmd("measure_temp")
    m = re.search(r"([0-9.]+)", txt or "")

    return float(m.group(1)) if m else None

def cpu_mhz() -> float | None:
    raw = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")

    try:
        return int(raw.strip()) / 1000 if raw else None
    
    except ValueError:
        return None

def throttled() -> str | None:
    txt = _vcgencmd("get_throttled")

    m = re.search(r"throttled=0x([0-9a-fA-F]+)", txt or "")

    if not m:
        return None
    
    bits = int(m.group(1), 16)

    if bits == 0:
        return "none"
    
    now = {0: "under-voltage", 1: "arm freq capped", 2: "currently throttled", 3: "soft temp limit"}
    past = {16: "under-voltage occurred", 17: "freq cap occurred", 18: "throttling occurred", 19: "soft temp limit occurred"}

    flags = [n for b, n in now.items() if bits & (1 << b)]
    flags += [n for b, n in past.items() if bits & (1 << b)]

    return ", ".join(flags) or f"0x{bits:x}"

class PowerSampler:
    def __init__(self, hz: float = 1.0):
        self.interval, self.samples, self._stop = 1.0 / hz, [], None

    def __enter__(self):
        if board_power_w() is None:
            return self # not a Pi 5; no-op
        
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

        return self

    def _run(self):
        while not self._stop.is_set():
            w = board_power_w()

            if w:
                self.samples.append((time.time(), w))

            self._stop.wait(self.interval)

    def __exit__(self, *exc):
        if self._stop:
            self._stop.set()
            self._t.join(timeout=2)

        return False

    def result(self) -> tuple[float | None, float | None]:
        """(mean watts, watt-hours)."""
        if len(self.samples) < 2:
            return (self.samples[0][1] if self.samples else None), None
        
        wh = 0.0

        for (t0, w0), (t1, w1) in zip(self.samples, self.samples[1:]):
            wh += (w0 + w1) / 2 * (t1 - t0) / 3600

        span = self.samples[-1][0] - self.samples[0][0]

        return (wh * 3600 / span if span > 0 else None), wh

def print_stats(dt, ttft, nframes, timings, pid, cpu0, pcpu0, power, ix, retr_ms):
    tm = timings or {}

    L = []

    idx = f"index {len(ix.docs):,} docs" if ix is not None else "no index"
    L.append(f"retrieval    {retr_ms:8.1f} ms    {idx}")

    p_n, p_ps = tm.get("prompt_n"), tm.get("prompt_per_second")

    if p_n:
        L.append(f"prompt       {p_n:8} tok   {p_ps:.1f} tok/s" + (f"    first token {ttft:.2f} s" if ttft else ""))
        
    elif ttft:
        L.append(f"first token  {ttft:8.2f} s")

    g_n, g_ps = tm.get("predicted_n"), tm.get("predicted_per_second")

    if g_n:
        L.append(f"generation   {g_n:8} tok   {g_ps:.1f} tok/s    wall {dt:.1f} s")

    else:
        L.append(f"generation   {nframes:8} frames  {nframes/max(dt,1e-9):.1f}/s" f"    wall {dt:.1f} s   (server sent no timings)")

    cpu1 = cpu_jiffies()
    ncpu = os.cpu_count() or 1

    if cpu0 and cpu1 and cpu1[1] > cpu0[1]:
        sys_pct = 100 * (cpu1[0] - cpu0[0]) / (cpu1[1] - cpu0[1])
        srv = ""
        pcpu1 = proc_cpu(pid) if pid else None

        if pcpu0 is not None and pcpu1 is not None:
            # percent of ONE core, so 400% is a fully-loaded Pi 5
            srv = f"    server {100*(pcpu1-pcpu0)/max(dt,1e-9):.0f}% of {100*ncpu}%"
        L.append(f"cpu          {sys_pct:7.0f}% system{srv}")

    cur, peak = proc_mem_mb(pid) if pid else (None, None)
    avail = mem_available_mb()

    if peak or avail:
        mem = f"memory       {peak:8.0f} MB peak" if peak else "memory       " + " " * 8

        if cur:
            mem += f"  ({cur:.0f} now)"

        if avail:
            mem += f"    {avail:.0f} MB available"
        L.append(mem)

    t, mhz, thr = soc_temp_c(), cpu_mhz(), throttled()

    if t or mhz or thr:
        bits = []

        if t:
            bits.append(f"{t:.1f} °C")

        if mhz:
            bits.append(f"{mhz:.0f} MHz")

        if thr and thr != "none":
            bits.append(f"THROTTLED: {thr}")

        elif thr:
            bits.append("not throttled")
        L.append("thermal      " + "   ".join(bits))

    mean_w, wh = power.result() if power else (None, None)

    if mean_w:
        line = f"energy       {mean_w:8.2f} W mean"

        if wh:
            line += f"   {wh*1000:.2f} mWh this answer"

            if tm.get("predicted_n"):
                line += f"   {wh*3600/tm['predicted_n']:.3f} J/tok"
        L.append(line + "   (SoC rails, not wall)")

    w = max(len(x) for x in L)
    print("  ┌" + "─" * (w + 2), file=sys.stderr)

    for x in L:
        print(f"  │ {x}", file=sys.stderr)

    print("  └" + "─" * (w + 2), file=sys.stderr)

# --------------------------------------------------------------------------- server
class Server:
    def __init__(self, model: Path, port: int, ctx: int, threads: int, kv_type: str = "q8_0", threads_batch: int | None = None):
        self.port, self.proc = port, None
        self.url = f"http://127.0.0.1:{port}"

        if self._alive():
            print(f"using llama-server already on :{port}", file=sys.stderr)

            return
        
        if not SERVER_BIN.exists():
            sys.exit(f"llama-server not found at {SERVER_BIN}\n" f"build it: cmake --build ~/llama.cpp/build -j4 --target llama-server")
            
        cmd = [str(SERVER_BIN), "-m", str(model), "-c", str(ctx), "-t", str(threads), "--port", str(port), "--host", "127.0.0.1", "-b", "128", "--jinja", "--no-warmup"]
        
        if kv_type and kv_type != "f16":
            cmd += ["-ctk", kv_type, "-ctv", kv_type]

        if threads_batch:
            cmd += ["-tb", str(threads_batch)]

        if not LITE:
            print(f"starting llama-server (ctx {ctx}, {threads} threads)…", file=sys.stderr)

        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        for _ in range(180):
            if self._alive():

                if not LITE:
                    print("server ready", file=sys.stderr)
                return
            
            if self.proc.poll() is not None:
                sys.exit("llama-server exited during startup — run it manually to see why")

            time.sleep(1)

        sys.exit("llama-server did not become ready in 180s")

    def _alive(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=2) as r:
                return r.status == 200
            
        except Exception:
            return False

    def stream(self, messages: list[dict], max_tokens: int, temp: float,
               prefill: str | None = None):
        # prefilling an assistant turn forces the model to continue from that text.
        # SFT trained on the bare question, so with a reference already in context
        # the model skips the reasoning -- it appeared only when retrieval FAILED
        if prefill:
            messages = messages + [{"role": "assistant", "content": prefill}]
        # timings_per_token attaches llama.cpp's own counters, so tok/s comes from
        # the server's token count rather than from counting SSE frames
        self.last_timings = None
        body = json.dumps({"messages": messages, "max_tokens": max_tokens, "temperature": temp, "top_p": 0.9, "stream": True, "timings_per_token": True, "repeat_penalty": 1.1}).encode()
        
        req = urllib.request.Request(f"{self.url}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
        
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()

                if not line.startswith("data:"):
                    continue

                payload = line[5:].strip()

                if payload == "[DONE]":
                    break

                try:
                    obj = json.loads(payload)
                    delta = obj["choices"][0].get("delta", {})

                except Exception:
                    continue

                if isinstance(obj.get("timings"), dict):
                    self.last_timings = obj["timings"]

                if delta.get("content"):
                    yield delta["content"]

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)

            try:
                self.proc.wait(timeout=10)

            except subprocess.TimeoutExpired:
                self.proc.kill()

# facets where OMITTING something is dangerous: quoted verbatim, model bypassed.
# asked for the metformin dose it dropped "do not use below eGFR 30"; for atorvastatin's
# side effects it emitted five fabricated statistics. `define` stays prose
VERBATIM_FACETS = {"dose", "contra", "warning", "sideeffect", "interact", "pregnancy"}

# brand -> generic. measured 0/8 without it: Lipitor, Tylenol, Advil and the rest
# all missed, since labels are indexed under the generic. built by
# scripts/fetcher.py drugs; an absent file just means no expansion
_ALIAS_FILE = ROOT / "artifacts" / "drug_aliases.json"

# international (INN) names for drugs the FDA indexes under US ones. openFDA never
# says "paracetamol", so that question was refused outright -- a vocabulary gap
# rather than a corpus one, and it affects every user outside the US
INN_NAMES = { "paracetamol": "acetaminophen",   "salbutamol": "albuterol", "adrenaline": "epinephrine",      "noradrenaline": "norepinephrine", "frusemide": "furosemide",        "lignocaine": "lidocaine", "amoxycillin": "amoxicillin",     "ciclosporin": "cyclosporine", "rifampicin": "rifampin",         "pethidine": "meperidine", "beclometasone": "beclomethasone", "thyroxine": "levothyroxine", "glyceryl trinitrate": "nitroglycerin", "bendroflumethiazide": "bendroflumethiazide", "indometacin": "indomethacin",    "oestradiol": "estradiol", "colecalciferol": "cholecalciferol", "chlorphenamine": "chlorpheniramine", "phenytoin sodium": "phenytoin",  "sodium valproate": "valproate", "co-codamol": "acetaminophen",    "hydroxycarbamide": "hydroxyurea", }

_TOPIC_ALIAS_FILE = ROOT / "artifacts" / "topic_aliases.json"

try:
    ALIASES: dict[str, str] = json.loads(_ALIAS_FILE.read_text())

except Exception:
    ALIASES = {}
# which keys came from openFDA brand names. only those get the document-frequency
# rule: a topic synonym is SUPPOSED to be ordinary language ("ptsd", "acid
# reflux"), and applying it there cost two of 71 realistic queries
_BRAND_KEYS: set[str] = set(ALIASES)

try:
    # MedlinePlus synonyms: "rubeola" -> "Measles". filtered at build time to drop
    # any that shadow a condition indexed in its own right
    ALIASES.update({k: v.lower() for k, v in json.loads(_TOPIC_ALIAS_FILE.read_text()).items()})

except Exception:
    pass

ALIASES.update(INN_NAMES)
_ALIAS_RE = (re.compile(r"\b(" + "|".join(sorted(map(re.escape, ALIASES), key=len, reverse=True)) + r")\b", re.I) if ALIASES else None)

def expand_aliases(text: str) -> str:
    if not _ALIAS_RE:
        return text
    
    return _ALIAS_RE.sub(lambda m: ALIASES[m.group(0).lower()], text)

def restore_possessives(text: str, ix: "BM25Index") -> str:
    if not getattr(ix, "poss", None):
        return text
    
    return re.sub(r"\b([A-Za-z]{3,}s)\b", lambda m: ix.poss.get(m.group(1).lower(), m.group(1)), text)

def prune_aliases(ix: "BM25Index") -> "BM25Index":
    global _ALIAS_RE

    indexed = {c for c in ix.dcond if c}
    shadowed = {k for k in ALIASES if k in indexed}

    # second rule: a one-word alias occurring in ordinary prose is not a brand -- 77
    # are ordinary english (`athletes -> terbinafine`). document frequency separates
    # them: brands in at most 6 documents, ordinary words 16-386. brand keys only
    shadowed |= {k for k in ALIASES if k in _BRAND_KEYS and len(ix.post_ids.get(k, ())) >= _ALIAS_MAX_DF}

    if shadowed:
        for k in shadowed:
            del ALIASES[k]

        _ALIAS_RE = (re.compile(r"\b(" + "|".join(sorted(map(re.escape, ALIASES), key=len,reverse=True)) + r")\b", re.I) if ALIASES else None)
    return ix

# verbatim sections bypass the model, so the context budget does not apply. capped
# only to keep one answer readable; the longest stored section is ~9,000 chars
VERBATIM_MAX = 9000

# some SOURCES are quoted regardless of facet: asked for neuroferritinopathy's symptoms
# the model applied the maximum row value (90%) to all ~30 when most are 7.5%. tested on
# the source, not the facet -- symptom prose paraphrases fine. 2,243 docs, 11.1%
VERBATIM_SOURCE = re.compile(r"Human Phenotype Ontology", re.I)

# preview one sentence of orientation before a quoted section, which otherwise answers a
# dose question with 2,624 median characters and no lead-in.
#
# the model composes it INSIDE A GRAMMAR: the kind and topic are literals, and its
# only choice is between two reason clauses. a wrong choice is still a true one
#
PREVIEW_KINDS = { "dose": "dosage and administration section", "contra": "contraindications section", "warning": "warnings and precautions section", "sideeffect": "adverse reactions section", "interact": "drug interactions section", "pregnancy": "pregnancy and lactation section", }

_TABLE_KIND = "table of reported signs and their individual frequencies"

# the section kind is supplied rather than asked for -- left to identify it from
# the text the model was unreliable, so no cross-check is claimed on it
PREVIEW_SYS = ( "You introduce a medical reference document that is shown to the reader in full " "beneath your sentence. Say what it is and why text of that kind is reproduced word " "for word rather than summarised. Do not describe its contents or state any number.")

def _gbnf(topic: str | None, is_table: bool, kind: str) -> str:
    # `kind` is a literal and `reason` is gated on is_table, both decided from the
    # source rather than offered to the model. letting it choose called a two-line
    # fluoxetine section "a table of reported signs", 1 in 30
    reasons = (['"the figures differ from line to line and one summary ' 'cannot stand for all of them"', '"averaging them would imply one number applies to every sign"'] if is_table else ['"leaving anything out of it could change what it says"', '"an omission in a section like this one can be dangerous"'])
    
    tp = f' " for {topic}"' if topic else '""'

    return (f'root ::= "This is the " kind{tp} ", quoted in full because " reason "."\n' f'kind ::= "{kind}"\n' f"reason ::= {' | '.join(reasons)}\n")

def verbatim_preview(question: str, text: str, is_table: bool, srv=None, port: int = 8080) -> str | None:
    topic = condition_of(question)

    if topic:
        topic = " ".join(w if w.isupper() else w.capitalize() for w in topic.split())

    expected = _TABLE_KIND if is_table else PREVIEW_KINDS.get(facet_of(question) or "")

    def fallback():
        if not expected:
            return None
        
        why = ("the figures differ from line to line and one summary cannot stand for " "all of them" if is_table else "leaving anything out of it could change what it says")
        
        return (f"This is the {expected}" + (f" for {topic}" if topic else "") + f", quoted in full because {why}.")

    if srv is None or not expected:
        return fallback()
    
    try:
        body = json.dumps({ "prompt": f"{PREVIEW_SYS}\n\nThe reader asked: \"{question}\"\n" f"They are about to be shown the {expected}" f"{', about ' + topic if topic else ''}. It opens:\n" f"{text[:400]}\n\nIntroduction:", "n_predict": 160, "temperature": 0.0, "grammar": _gbnf(topic, is_table, expected)}).encode()
        
        req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body, headers={"Content-Type": "application/json"})
        
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read())["content"].strip()

    except Exception:
        return fallback()
    
    if not out.startswith("This is the ") or not out.endswith("."):
        return fallback()
    
    return out

def trim(text: str, limit: int, note: bool = True) -> str:
    if len(text) <= limit:
        return text
    
    # `note` off for model context: the marker is for the READER. Pasted into reference
    # material it is just a sentence the model may copy into its answer.
    tail = "\n\n[section continues — consult the full label]" if note else ""
    head = text[:limit]

    for sep in (". ", "; ", ", "):
        i = head.rfind(sep)

        if i > limit * 0.6:
            return head[:i + 1].rstrip() + tail
        
    return head.rsplit(" ", 1)[0].rstrip() + " …" + tail

def pretty_condition(c: str) -> str:
    c = re.sub(r"^what i need to know about\s+", "", c)
    c = re.sub(r"^(do you have information about|learning about)\s+", "", c)

    return c.strip()

def narrower_matches(ix: "BM25Index", qc: str, facet: str, limit: int = 8) -> list[str]:
    seen: list[str] = []

    for d, dc in enumerate(ix.dcond):

        if dc and dc != qc and qc in dc and ix.dfacet[d] == facet:
            label = pretty_condition(dc)
            # after cleaning, the booklet title collapses onto the plain condition -- so
            # it is no longer a distinct option, and would just repeat the question

            if label and label != qc and label not in seen:
                seen.append(label)

                if len(seen) >= limit:
                    break
    return seen

# a misspelling scores as MAXIMALLY off-domain: coverage weights an unknown term at
# max_idf. "what are the syntoms of gout?" ranked the right entry first at 28.4 and
# still came to 0.36 coverage. the discriminator is DOCUMENT FREQUENCY, not string
# distance, and it suggests without rewriting -- `hypertention` fits two opposites
_MIN_DF_SUGGEST = 25          # below this the candidate is probably a typo in the corpustoo: `symtoms` and `symtpoms` are both in the vocabulary

def did_you_mean(ix: "BM25Index", question: str, limit: int = 2) -> list[str]:
    out: list[str] = []

    for term in tokenize(question):

        if term in ix.idf or len(term) < 5:
            continue

        # Narrow before comparing: same initial, length within two, and common enough to
        # be a real word. Comparing against all 32k terms gets the same answer slower.
        cands = [v for v in ix.idf
                 # length within ONE, not two. "syntoms"->"symptoms" and
                 # "torque"->"true" both score 0.80 on similarity, but a real typo
                 # barely changes length while "torque"/"true" differ by two
                 if v[:1] == term[:1] and abs(len(v) - len(term)) <= 1
                 and len(ix.post_ids.get(v, ())) >= _MIN_DF_SUGGEST]
        
        near = difflib.get_close_matches(term, cands, n=4, cutoff=0.78)
        near.sort(key=lambda v: -len(ix.post_ids.get(v, ())))

        if near:
            out.append(f"{term} → {' or '.join(near[:limit])}")

    return out

def _idf_overlap(ix: BM25Index, want: set[str], have: set[str]) -> float:
    mx = max(ix.idf.values()) if ix.idf else 1.0
    total = sum(ix.idf.get(w, mx) for w in want)

    return sum(ix.idf.get(w, mx) for w in want if w in have) / total if total else 0.0

def build_messages(question: str, ix: BM25Index | None, min_score: float, max_ctx_chars: int, verbose: bool,  min_coverage: float = 0.5, min_topic: float = 0.4, _retry: bool = True) -> tuple[list[dict], bool, list[str], str | None]:
    if ix is None:
        return ([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}], False, [], None)

    def refused_retry():
        if not _retry:
            return None
        
        bare = re.sub(r"\b(?:an?|the)\s+", "", question)

        if bare == question:
            return None
        
        m2, g2, o2, v2 = build_messages(bare, ix, min_score, max_ctx_chars, False, min_coverage, min_topic, _retry=False)

        if v2 or (g2 and not o2):

            if verbose:
                print(f"  [retried without articles: {bare!r}]", file=sys.stderr)

            return (m2, g2, o2, v2)
        
        return None

    def refuse(user_msg: str, options: tuple[str, ...] | list[str] = ()):
        return refused_retry() or ([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}], False, list(options), None)
    
    # Retrieval view of the question: brand names resolved to generics and possessives
    # restored. Display and prompt text keep the user's own wording.
    rq = restore_possessives(expand_aliases(question), ix)
    hits = ix.search(rq, k=1)

    # raw BM25 is uncalibrated across queries: a bicycle-torque question matched
    # "Pili torti" at 10.9, above an 8.0 floor. coverage is the real gate, and it is
    # IDF-weighted -- plain counting passed "what cause guat?" at 0.50
    q_terms = set(tokenize(rq))
    coverage, score, doc = 0.0, (hits[0][0] if hits else 0.0), (hits[0][1] if hits else -1)

    if hits and q_terms:
        d_terms = set(tokenize(ix.docs[doc][0] + " " + ix.docs[doc][1]))
        coverage = _idf_overlap(ix, q_terms, d_terms)

    if not hits or score < min_score or coverage < min_coverage:

        if verbose:
            print(f"  [no usable reference — score {score:.1f} (need {min_score}), " f"coverage {coverage:.2f} (need {min_coverage})]", file=sys.stderr)
            
        user = (f"Question: {question}\n\n" "IMPORTANT: no reference material was found for this question in your " "medical corpus. Do NOT guess. If the question is outside medicine, say " "it is outside your scope. If it is medical but unfamiliar, say you do " "not have a reference for it and recommend consulting a clinician. " "Give no numbers, dosages, or specifics.")
        # A spelling suggestion is offered, not applied -- see did_you_mean(). It rides the
        # existing options channel, which already exists for the disambiguation path.
        typos = did_you_mean(ix, question)

        return refuse(user, [f"did you mean:  {t}" for t in typos])

    qc_, dc_ = condition_of(rq), (ix.dcond[doc] if ix.dcond else "")
    qf_, df_ = facet_of(rq), (ix.dfacet[doc] if ix.dfacet else "")

    # FACET GAP: right condition, wrong aspect, nothing covering the aspect asked.
    # "symptoms of cancer" matched the DEFINITION entry, so the condition check below
    # stayed silent and the model invented a 22-item symptom list
    if qc_ and dc_ and qc_ == dc_ and qf_ != "other" and qf_ != df_:
        has_facet = any(c == qc_ and f == qf_ for c, f in zip(ix.dcond, ix.dfacet))

        if not has_facet:
            # before refusing: does a MORE SPECIFIC condition cover this aspect?
            # MedQuAD splits by population, so "what causes kidney stones?" was refused
            # with the answer in the index -- 301 pairs unreachable. offering them is
            # also honest: causes in children are not causes in adults
            opts = narrower_matches(ix, qc_, qf_)

            if len(opts) > 1:

                if verbose:
                    print(f"  [no {qf_!r} entry for {qc_!r} itself — offering " f"{len(opts)} more specific ones]", file=sys.stderr)
                    
                return ([], False, opts, None)
            
            if verbose:
                print(f"  [corpus has {qc_!r} but nothing on its {qf_!r} aspect]", file=sys.stderr)
                
            user = (f"Question: {question}\n\n" f"IMPORTANT: the reference corpus covers {qc_} but has no entry on " f"that specific aspect of it. Do NOT answer from memory. Say plainly " f"that you do not have a reference for this aspect and suggest " f"asking about a more specific condition.")
            
            return refuse(user)

    # Asked about something broader than anything indexed -> disambiguate, do not
    # substitute a narrower topic and present it as the answer.
    if qc_ and dc_ and qc_ != dc_ and qc_ in dc_:
        opts = narrower_matches(ix, qc_, facet_of(rq))

        if len(opts) > 1:

            if verbose:
                print(f"  [no entry for {qc_!r} itself — offering {len(opts)} specific ones]", file=sys.stderr)

            return ([], False, opts, None)

    # TOPIC MATCH (reverse coverage): does the document's subject appear in the question?
    # coverage asks only the forward question, which a long document satisfies by accident
    # -- "vitamin deficiency" hit a hyperparathyroidism entry at 1.00 and got its dosing
    if dc_:
        topic = _idf_overlap(ix, set(tokenize(dc_)), q_terms)

        if topic < min_topic:
            if verbose:
                print(f"  [best match is about {dc_!r}, which the question does not " f"mention (topic match {topic:.2f})]", file=sys.stderr)
                
            user = (f"Question: {question}\n\n" f"IMPORTANT: the closest reference is about {dc_}, which is not what " f"was asked. Do NOT answer from it or from memory. Say you have no " f"suitable reference and ask for a more specific question.")
            
            return refuse(user)

    q, a = ix.docs[doc]

    # Quote when the FACET is safety-critical, or when the SOURCE is a frequency table
    # this model cannot read (see VERBATIM_SOURCE).
    table_source = bool(VERBATIM_SOURCE.search(a))

    if qf_ in VERBATIM_FACETS or table_source:
        why = qf_ if qf_ in VERBATIM_FACETS else "frequency table"

        if verbose:
            print(f"  {S.GREY}[ref: {q[:60]!r} score {score:.1f}]{S.RESET} " f"{S.GREEN}verbatim ({why}){S.RESET}{S.GREY} — not paraphrased{S.RESET}", file=sys.stderr)
            
        return ([], True, [], f"{q}\n\n{trim(a, VERBATIM_MAX)}")

    if verbose:
        note = ""

        qc, dc = condition_of(rq), (ix.dcond[doc] if ix.dcond else "")

        if qc and dc and qc != dc:
            note = (f"  !! asked about {qc!r}, best match covers {dc!r}" if qc in dc or dc in qc else f"  !! condition mismatch: {qc!r} vs {dc!r}")
            
        print(f"  [ref: {q[:60]!r} score {score:.1f} coverage {coverage:.2f}]{note}", file=sys.stderr)
        
    ctx = trim(f"{q}\n{a}", max_ctx_chars, note=False)
    user = f"Reference material:\n{ctx}\n\nQuestion: {question}"

    return ([{"role": "system", "content": SYSTEM_PROMPT},{"role": "user", "content": user}], True, [], None)

def main() -> None:
    ap = argparse.ArgumentParser(description="EXCALIBUR-Medic with BM25 retrieval")
    ap.add_argument("-q", "--question", help="answer one question and exit")
    ap.add_argument("-m", "--model", type=Path, default=DEFAULT_MODEL, help="GGUF to serve; defaults to the promoted Q4_K_M")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="corpus to index; defaults to the combined CSV, built on " "first use from MedQuAD + openFDA + MedlinePlus")
    ap.add_argument("--no-rag", action="store_true", help="ablation: disable retrieval")
    ap.add_argument("--rebuild-index", action="store_true", help="ignore the cached index even if its corpus stamp matches")
    ap.add_argument("--min-score", type=float, default=2.0, help="BM25 floor. Deliberately LOW: raw BM25 scales with term rarity, " "so an absolute threshold rejects common conditions (idf 'cancer' " "2.32 -> score 7.4) while passing irrelevant rare ones (a bicycle " "question scored 12.2). Coverage is the real gate; this only " "catches near-zero matches.")
    ap.add_argument("--min-coverage", type=float, default=0.5, help="fraction of question terms that must appear in the retrieved doc")
    ap.add_argument("--min-topic", type=float, default=0.4, help="how much of the retrieved document's own subject must appear in " "the question. Catches bare keyword queries that match a long " "document by accident.")
    ap.add_argument("--max-ctx-chars", type=int, default=2600, help="reference characters injected; raised with the 2048 ctx default")
    ap.add_argument("-c", "--ctx", type=int, default=2048, help="Measured on a Pi 5 2GB: 2048 ctx with a q8_0 KV cache peaks at " "1638 MB, versus 1636 MB for 1024 ctx with f16 KV. Doubling the " "context is effectively free, and it stops long references and " "answers being truncated.")
    ap.add_argument("--kv-type", default="q8_0", choices=["f16", "q8_0", "q4_0"], help="KV cache precision. q8_0 halves KV memory, which is what pays " "for the larger context.")
    ap.add_argument("-t", "--threads", type=int, default=2, help="GENERATION threads. Measured on a Pi 5: 2 threads gives 14.6 " "tok/s vs 12.8 at 4 -- generation is memory-bandwidth bound, so " "extra cores contend rather than help.")
    ap.add_argument("-tb", "--threads-batch", type=int, default=4, help="PROMPT-PROCESSING threads. This half IS compute-bound and does " "scale: 78.7 tok/s at 4 threads vs 44.6 at 2. Splitting the two " "is worth ~10%% end-to-end on a RAG prompt (44.4s vs 49.2s for " "800 prompt + 500 generated tokens).")
    ap.add_argument("-n", "--max-tokens", type=int, default=500, help="generation cap; verbatim answers bypass it entirely")
    ap.add_argument("--temp", type=float, default=0.0, help="0.0 (greedy) is the default and matters more than it looks. At " "0.3 the model dropped colchicine, allopurinol, febuxostat and " "probenecid from a gout answer whose reference listed all four; " "at 0.0 it recovered 7/8 target drugs. Sampling picks a shorter " "path early and cannot recover. Greedy also makes answers " "reproducible, which a reference tool wants. Measured no " "repetition degeneration (distinct-4gram 0.87-1.00).")
    ap.add_argument("--port", type=int, default=8080, help="llama-server port; an existing server here is reused")
    ap.add_argument("--allow-ungrounded", dest="strict", action="store_false", help="let the model answer with no reference (it will fabricate)")
    ap.set_defaults(strict=True)
    ap.add_argument("--quiet", action="store_true", help="hide retrieval diagnostics")
    ap.add_argument("--no-reasoning", dest="reasoning", action="store_false", help="skip the <reasoning> block and answer directly — about half " "the tokens, which matters on a Pi")
    ap.set_defaults(reasoning=True)
    ap.add_argument("--lite", action="store_true", help="skip the startup panel and go straight to the prompt")
    ap.add_argument("--color", default="auto", choices=["auto", "always", "never"], help="auto disables colour when stdout is not a terminal, so piping " "into a log file (as benchmark_pi.sh does) stays clean. NO_COLOR " "is honoured")
    ap.add_argument("--stats", action="store_true", help="per-answer compute/memory/thermal/energy block. Energy needs a " "Pi 5 (PMIC via vcgencmd); everything else degrades quietly")
    
    args = ap.parse_args()
    S.enable(args.color)

    global LITE
    LITE = args.lite

    if LITE:
        print("  loading…", end="\r", file=sys.stderr, flush=True)

    if not args.model.exists():
        sys.exit(f"model not found: {args.model}")

    ix = None

    if not args.no_rag:
        
        if not args.csv.exists():
            sys.exit(f"corpus not found: {args.csv} (needed for retrieval; or use --no-rag)")

        ix = get_index(args.csv, INDEX_CACHE, args.rebuild_index)

    srv = Server(args.model, args.port, args.ctx, args.threads, args.kv_type, args.threads_batch)
    verbose = not args.quiet

    REFUSAL = ( "<answer>\nI do not have a reference for that in my medical corpus, so I cannot " "answer it reliably. If this is a clinical question, please consult a clinician " "or a current medical reference.\n</answer>")

    def answer(qs: str) -> None:
        _r0 = time.time()
        msgs, grounded, options, verbatim = build_messages(qs, ix, args.min_score, args.max_ctx_chars, verbose, args.min_coverage, args.min_topic)

        retr_ms = (time.time() - _r0) * 1000

        if verbatim:
            is_table = bool(VERBATIM_SOURCE.search(verbatim))

            _pv = verbatim_preview(qs, verbatim, is_table, srv, args.port)

            heading = 'QUOTED FROM SOURCE' if is_table else 'QUOTED FROM LABEL'

            print(f"\n{rule(heading + ' — not summarised', colour=S.GREEN)}")
            print(f"{S.GREY}{_pv}{S.RESET}\n")
            print(verbatim)

            if is_table:
                print(f"\n{S.YELLOW}Frequencies are per sign, not for the condition as a " f"whole.{S.RESET}{S.GREY} They come from\nstudies of small numbers " f"of patients and are rough estimates. Not advice for you." f"{S.RESET}")
                
            else:
                print(f"\n{S.YELLOW}This is general labelling, not advice for you." f"{S.RESET}{S.GREY} Doses depend on kidney and liver\nfunction, " f"age, weight, pregnancy and other medicines. Check with a " f"clinician\nor pharmacist.{S.RESET}")
                
            print(rule('', colour=S.GREEN))

            return
        
        if options:
            spell = [o for o in options if o.startswith("did you mean:")]

            if spell:
                print(f"\n{rule('CHECK THE SPELLING', colour=S.YELLOW)}")

                print("One word is close to a common one in the corpus:\n")

                for o in spell:
                    print(f"  {S.CYAN}•{S.RESET} {o[len('did you mean:'):].strip()}")

                print(f"\n{S.GREY}Not corrected automatically — a single edit can change " f"which condition is\nmeant, and two of them can be opposites." f"{S.RESET}")
                
                print(rule('', colour=S.YELLOW))

            else:
                qc_ = condition_of(qs)

                print(f"\n{rule('WHICH ONE?', colour=S.YELLOW)}")
                print(f"No general entry for {S.BOLD}{qc_}{S.RESET}, but these are " f"indexed:\n")
                
                for o in options:
                    print(f"  {S.CYAN}•{S.RESET} {o}")

                print(rule('', colour=S.YELLOW))
                
                return
            
        if not grounded and ix is not None and args.strict:
            _s = TagStyler()
            _s.feed(REFUSAL)
            _s.flush()

            if verbose:
                print("  [refused at application layer]", file=sys.stderr)

            return
        
        pid = srv.proc.pid if getattr(srv, "proc", None) else None
        cpu0 = cpu_jiffies()
        pcpu0 = proc_cpu(pid) if pid else None
        t0, ntok, ttft = time.time(), 0, None

        styler = TagStyler()

        with PowerSampler() as power:
            for chunk in srv.stream(msgs, args.max_tokens, args.temp, prefill="<reasoning>\n" if args.reasoning else None):
                
                if ttft is None:
                    ttft = time.time() - t0

                styler.feed(chunk); ntok += 1

        styler.flush()
        dt = time.time() - t0

        if verbose:
            tag = "grounded in reference" if grounded else "UNGROUNDED — no reference found"
            tm = getattr(srv, "last_timings", None) or {}
            gen = tm.get("predicted_per_second")
            ntok_real = tm.get("predicted_n", ntok)
            rate = f"{gen:.1f} tok/s" if gen else f"{ntok/max(dt,1e-9):.1f} frames/s"
            
            print(f"  [{tag} · {ntok_real} tok · {rate}]", file=sys.stderr)

        if args.stats:
            print_stats(dt, ttft, ntok, getattr(srv, "last_timings", None), pid, cpu0, pcpu0, power, ix, retr_ms)

    try:
        if args.question:
            answer(args.question)

            return
        
        if args.lite:
            print(" " * 20, end="\r", file=sys.stderr, flush=True)   # erase "loading…"

        else:
            startup_report(args, ix)

        while True:
            try:
                q = input(f"{S.BOLD}{S.CYAN}You ▸ {S.RESET}").strip()

            except (EOFError, KeyboardInterrupt):
                print(); break
            
            if not q:
                continue

            if q.lower() in {"exit", "quit"}:
                break

            if q in {"?", "help"}:
                print(f"{S.GREY}  ?            this help\n" f"  stats        toggle the per-answer compute block\n" f"  exit / quit  leave\n" f"  anything else is treated as a question{S.RESET}\n")
                
                continue

            if q.lower() == "stats":
                args.stats = not args.stats
                print(f"{S.GREY}  stats {'on' if args.stats else 'off'}{S.RESET}\n")

                continue
            
            answer(q)
            print()

    finally:
        srv.close()

if __name__ == "__main__":
    main()