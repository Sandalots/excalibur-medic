#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, re, sys, gc
from pathlib import Path

# never open a window; this is a command-line tool
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbformat, numpy as np, mlx.core as mx
from mlx_lm import batch_generate, generate
from mlx_lm.sample_utils import make_sampler
from mlx.utils import tree_unflatten
import excalibur_inference as MR

ANS = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
# format compliance means both tags in order -- ANS above would pass a bare
# <answer>, and the finding is that the base model emits neither
TAGGED = re.compile(r"<reasoning>(.*?)</reasoning>\s*<answer>(.*?)</answer>", re.S | re.I)
REFUSED = re.compile(r"do not have a reference|cannot answer it reliably|" r"does not cover|outside (my|what)", re.I)

# --------------------------------------------------------------- measured constants
# results one notebook run cannot produce: a second model, several training runs,
# or the Pi. defined here so a figure and its source cannot disagree silently.
# each entry names how to regenerate it -- update a block whole, never one number
MEASURED = {
    # `evaluate.py ablate` -- 40 held-out questions, identical retrieved reference,
    # greedy decoding, only the weights differing
    "ablation": {
        "labels":    ["base\nLlama-3.2-1B", "shipped adapter\nSFT + behaviour"],
        "grounding": [0.473, 0.602],
        "format":    [0.000, 0.975],
    },

    # grounding floors on 600 MedQuAD rows. The oracle row is `evaluate.py ceiling`;
    # the others score generated answers against deliberately wrong references
    "grounding_floors": [
        ("null — a different condition entirely",  0.090),
        ("right condition, wrong question type",   0.221),
        ("unaided (no retrieval)",                 0.317),
        ("WITH RETRIEVAL — shipped",               0.705),
        ("oracle (perfect retrieval)",             0.716),
        ("ceiling — answer scored against itself", 1.000),
    ],

    # teacher pilots: 120 questions, same quality gate, same patches
    # (name, mean answer words, grounding, % kept by the gate)
    "teachers": [
        ("Bio-Medical-Llama-3-8B",  53, 0.706,  99.2),
        ("OpenBioLLM-8B (shipped)", 62, 0.675,  92.5),
        ("Qwen2.5-14B-Instruct",    68, 0.582, 100.0),
        ("HuatuoGPT-o1-7B",         79, 0.571,  50.8),
        ("II-Medical-8B",          269, 0.321,   1.7),
    ],

    # strict top-1 on 1,200 held-out, measured on the MedQuAD-ONLY corpus of the
    # time. not comparable to today's -- re-running gives a different ladder
    "retrieval_ladder": [
        ("grounding metric's stop list (the bug)", 49.2),
        ("retrieval-specific stop list",           55.0),
        ("+ question field weight ×6",             58.8),
        ("+ facet boost",                          79.3),
        ("+ condition boost  (shipped)",           88.2),
    ],

    # what the behaviour corpus bought, before and after mixing it into the SFT set
    "behaviour": {
        "format":     [0.80, 1.00],     # held-out format compliance
        "length":     [585, 368],       # mean answer length, characters
        "abstention": [16 / 16, 0 / 40],# refused when unanswerable / false refusals
    },

    # `benchmark_pi.sh` on the deployed Pi 5, 2 GB, headless
    "pi_threads": {
        "threads":    [1, 2, 3, 4],
        "generation": [9.6, 14.6, 13.7, 12.7],
        "prompt":     [21.9, 44.3, 63.4, 77.7],
    },

    # three separate SFT runs at increasing data volume. Perplexity is each run's best
    # validation figure; grounding is its held-out unaided mean
    "capacity": {
        "samples":    [162, 481, 4900],
        "perplexity": [3.14, 2.54, 2.59],
        "grounding":  [0.30, 0.30, 0.31],
    },
}

# Pi measurement log readers
# device figures are read back from the logs benchmark_pi.sh wrote on the Pi
# both readers return None when absent, so a checkout without them still runs
MEASUREMENTS = ROOT / "measurements"

# pull one number out of a log, or None
def _num(pat, text, cast=float, default=None):
    m = re.search(pat, text)

    return cast(m.group(1)) if m else default

# newest benchmark log, or None on a checkout without one
def find_bench(root: Path = MEASUREMENTS):
    c = sorted(Path(root).glob("benchmark_pi_*.txt"))

    return c[-1] if c else None

# newest energy report, or None on a checkout without one
def find_energy(root: Path = MEASUREMENTS):
    c = sorted(Path(root).glob("energy_*/report.txt"))

    return c[-1] if c else None

# thread sweep, peak memory, index cost and the end-to-end pass
def read_bench(path) -> dict:
    t = Path(path).read_text()

    return {
        "source":     Path(path).name,
        "cores":      _num(r"cores:\s*(\d+)", t, int),
        "ram_total":  _num(r"RAM:\s*(\d+)\s*MB total", t, int),
        "ram_avail":  _num(r"total,\s*(\d+)\s*MB available", t, int),
        # threads / gen tok/s / prompt tok/s / total s
        "threads":    [(int(a), float(b), float(c), float(d)) for a, b, c, d in
                       re.findall(r"^(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", t, re.M)],
        # ctx / KV type / peak RSS MB / free MB
        "memory":     [(int(x), kv, int(r), int(f)) for x, kv, r, f in
                       re.findall(r"ctx\s+(\d+)\s+KV\s+(\S+)\s+\([^)]*\)\s+peak RSS\s+(\d+)\s+MB\s+free while up:\s+(\d+)", t)],
        "docs":       _num(r"docs\s+(\d+)\s+terms", t, int),
        "terms":      _num(r"terms\s+(\d+)", t, int),
        "build_s":    _num(r"build\s+([\d.]+)s", t),
        "query_ms":   _num(r"query\s+([\d.]+)\s*ms", t),
        "retr_ms":    _num(r"retrieval\s+([\d.]+)\s*ms", t),
        "first_tok":  _num(r"first token\s+([\d.]+)\s*s", t),
        "wall_s":     _num(r"wall\s+([\d.]+)\s*s", t),
        "gen_tps":    _num(r"generation\s+\d+\s+tok\s+([\d.]+)\s*tok/s", t),
        "prompt_tps": _num(r"prompt\s+\d+\s+tok\s+([\d.]+)\s*tok/s", t),
    }

# power per phase, joules per query, and whether the soak throttled
def read_energy(path) -> dict:
    t = Path(path).read_text()
    phases = {}

    for label, key in (("board idle, no server", "idle_bare"),
                       ("server resident, idle", "idle_server"),
                       ("sustained generation",  "generating")):
        m = re.search(rf"{re.escape(label)}\s+([\d.]+)\s*W mean\s+([\d.]+)\s*W peak"
                      rf"\s+([\d.]+)\s*°C peak\s+\((\d+) samples\)", t)

        if m:
            phases[key] = dict(mean=float(m.group(1)), peak=float(m.group(2)),
                               temp=float(m.group(3)), samples=int(m.group(4)))

    # two numbers off one line, or None if the line is absent
    def pair(pat):
        m = re.search(pat, t)

        return tuple(float(x) for x in m.groups()) if m else None

    guardfree = re.search(r"(\d+) of (\d+) questions returned in ([\d.]+)-([\d.]+) s", t)

    return {
        "source":        Path(path).parent.name,
        "phases":        phases,
        "resident_cost": _num(r"holding the model resident costs\s+(-?[\d.]+)\s*W", t),
        "gen_cost":      _num(r"generating costs\s+\+?([\d.]+)\s*W", t),
        # (answered without the model, of total, fastest s, slowest s)
        "no_model":      tuple(float(g) for g in guardfree.groups()) if guardfree else None,
        "n_queries":     _num(r"PER QUERY\s+\(n=(\d+)", t, int),
        "latency_s":     _num(r"median latency\s+([\d.]+)\s*s", t),
        "gross_j":       _num(r"median energy, gross\s+([\d.]+)\s*J", t),
        "marginal_j":    _num(r"median energy, MARGINAL\s+([\d.]+)\s*J", t),
        "range_j":       pair(r"range, marginal\s+([\d.]+)\s*-\s*([\d.]+)\s*J"),
        "n_gens":        _num(r"SUSTAINED LOAD\s+\((\d+) generations", t, int),
        "soak_min":      _num(r"generations over (\d+) min", t, int),
        "first":         pair(r"first\s+\d+ generations\s+([\d.]+) tok/s\s+([\d.]+)"),
        "last":          pair(r"last\s+\d+ generations\s+([\d.]+) tok/s\s+([\d.]+)"),
        "change_tps":    _num(r"change\s+([+-][\d.]+) tok/s", t),
        "throttled":     not re.search(r"get_throttled\s+clean", t),
        "duty_qph":      _num(r"AT (\d+) QUERIES/HOUR", t, int),
        "duty_w":        _num(r"average draw\s+([\d.]+)\s*W", t),
    }

# both logs, each None when missing, so the notebook degrades quietly
def load_pi_measurements() -> tuple[dict | None, dict | None]:
    b, e = find_bench(), find_energy()

    return (read_bench(b) if b else None), (read_energy(e) if e else None)

# how much of the answer appears in the reference -- the grounding metric
def ground(answer: str, reference: str) -> float:
    # content words only, so stopwords cannot inflate the overlap
    a = {w for w in re.findall(r"[a-z]{4,}", answer.lower()) if w not in MR.STOP}
    r = set(re.findall(r"[a-z]{4,}", reference.lower()))

    return len(a & r) / max(len(a), 1)

# bootstrap confidence interval over the paired differences
def boot_ci(d: np.ndarray, n: int = 5000, seed: int = 0):
    # resample the paired differences to get a confidence interval
    rng = np.random.default_rng(seed)

    return np.percentile([rng.choice(d, len(d), replace=True).mean() for _ in range(n)], [2.5, 97.5])

# every code cell of the notebook, source only
def nb_cells() -> list[str]:
    nb = nbformat.read(ROOT / "excalibur.ipynb", as_version=4)

    return [c.source for c in nb.cells if c.cell_type == "code"]

# the one cell holding a marker. select by CONTENT, not index -- a rename and inserted
# cells once made both the filename and every index wrong. markers survive editing
def cell(cc: list[str], marker: str) -> str:
    hits = [c for c in cc if marker in c]

    assert len(hits) == 1, f"{marker!r} matched {len(hits)} cells, expected 1"

    return hits[0]

# load the notebook's imports and config into a namespace this script can call
def setup(n_prompts: int):
    cc = nb_cells()
    cell_ = lambda m: cell(cc, m)

    # notebook cells run under IPython, which supplies display(); a plain exec does
    # not, and without it the data cell died and left `ns` half built. the calls are
    # decorative, so a no-op will do -- but it has to exist
    # RUN_INSTALL=False so rebuilding eval state never reaches for the network
    ns = {"__name__": "__main__", "display": lambda *a, **k: None, "RUN_INSTALL": False}

    # imports and configuration are separate cells; both markers assert uniqueness
    exec(cell_("import mlx.core as mx"), ns)
    exec(cell_("TEACHER_REPO  ="), ns)
    exec(cell("df = pd.read_csv(MEDQUAD)"), ns)

    syn = ns["SYNTH"]
    
    seed_data = [json.loads(l) for l in open(syn / "sft_val.jsonl")]
    ns["sft_train"] = ns["sft_val"] = seed_data

    exec(cell("student, tok = load(STUDENT_REPO)"), ns)
    exec(cell("def render(question"), ns)

    df = ns["df"]

    test = df[df.split == "test"].sample(min(200, (df.split == "test").sum()), random_state=1337)
    rows = [test.iloc[i] for i in range(min(n_prompts, len(test)))]

    return ns, rows

# swap adapters without rebuilding the rest of the state
def load_adapter(ns, name: str):
    w = mx.load(str(ns["CKPT"] / f"{name}.safetensors"))
    ns["student"].update(tree_unflatten(list(w.items())))

    mx.eval(ns["student"].parameters()); ns["student"].eval()

# one generation, greedy, with or without a reference
def ask(ns, question: str, context: str | None, max_tokens: int = 450) -> str:
    user = question if not context else f"Reference material:\n{context}\n\nQuestion: {question}"
    p = ns["tok"].apply_chat_template([{"role": "system", "content": ns["SYS_STUDENT"]}, {"role": "user", "content": user}], add_generation_prompt=True)
    
    return generate(ns["student"], ns["tok"], p, max_tokens=max_tokens, sampler=make_sampler(temp=0.0), verbose=False)

# what the device would retrieve for this question
def retrieve(ix, question: str, chars: int = 1600):
    hits = ix.search(question, k=1)

    if not hits:
        return None, None
    
    doc = hits[0][1]
    q, a = ix.docs[doc]

    return f"{q}\n{a[:chars]}", doc

# n adapters on the same prompts, paired, with a bootstrap ci
def cmd_compare(args):
    ns, rows = setup(args.n)
    ix = MR.get_index(ROOT / "datasets" / "medquad.csv", MR.INDEX_CACHE, False) if not args.no_rag else None
    
    res = {}

    for name in args.adapters:
        load_adapter(ns, name)
        g, refused = [], 0

        for r in rows:
            ctx = retrieve(ix, r.question)[0] if ix else None
            out = ask(ns, r.question, ctx)
            
            m = ANS.search(out)
            a = (m.group(1) if m else out).strip()

            if REFUSED.search(a):
                refused += 1

            g.append(ground(a, r.answer))

        res[name] = np.array(g)
        answered = res[name][res[name] > 0]

        print(f"{name:<22} grounding {res[name].mean():.3f}  "
              f"median {np.median(res[name]):.3f}  refused {refused}/{len(rows)}  "
              f"grounding-when-answered {answered.mean() if len(answered) else 0:.3f}",
              flush=True)
        
    names = list(res)

    for i in range(1, len(names)):
        d = res[names[i]] - res[names[0]]
        lo, hi = boot_ci(d)

        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not significant"

        print(f"\n{names[i]} - {names[0]}: {d.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"{sig}  ({(d > 0).sum()}/{len(d)} better)")
        
    if len(names) > 1:
        print("\nNOTE: a correct refusal scores 0 here, same as a wrong answer. If one adapter refuses, run `refusal` before concluding it is worse.")

# does it refuse when retrieval missed, and answer when it did not?
def cmd_refusal(args):
    ns, rows = setup(args.n)
    ix = MR.get_index(ROOT / "datasets" / "medquad.csv", MR.INDEX_CACHE, False)
    
    key = {}

    for i, (q, a) in enumerate(ix.docs):
        key.setdefault((q.strip(), a.strip()[:120]), i)

    load_adapter(ns, args.adapter)

    tab = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}

    for r in rows:
        gold = key.get((r.question.strip(), r.answer.strip()[:120]))
        ctx, doc = retrieve(ix, r.question)

        if doc is None or gold is None:
            correct = False

        elif args.strict:
            correct = (doc == gold)

        else:
            same_cond = (MR.condition_of(ix.docs[doc][0]) == MR.condition_of(ix.docs[gold][0]) != None)
            same_facet = MR.facet_of(ix.docs[doc][0]) == MR.facet_of(ix.docs[gold][0])

            correct = bool(doc == gold or (same_cond and same_facet))

        out = ask(ns, r.question, ctx)
        tab[(correct, bool(REFUSED.search(out)))] += 1

    hit = tab[(True, True)] + tab[(True, False)]
    miss = tab[(False, True)] + tab[(False, False)]

    print(f"\n{'':22}{'REFUSED':>10}{'ANSWERED':>10}")
    print(f"{'retrieval CORRECT':22}{tab[(True, True)]:>10}{tab[(True, False)]:>10}")
    print(f"{'retrieval MISSED':22}{tab[(False, True)]:>10}{tab[(False, False)]:>10}")

    mode = "exact-row" if args.strict else "functional (same condition + facet)"

    print(f"\nretrieval accuracy      : {100*hit/max(hit+miss,1):.1f}%  [{mode}]")
    print(f"refuses when MISSED     : {100*tab[(False,True)]/max(miss,1):.0f}%  <- want HIGH")
    print(f"refuses when CORRECT    : {100*tab[(True,True)]/max(hit,1):.0f}%  <- want LOW")

# student against teacher, with and without the reference
def cmd_ceiling(args):
    ns, rows = setup(args.n)
    load_adapter(ns, args.adapter)

    s = np.array([ground((ANS.search(o).group(1) if ANS.search(o) else o).strip(), r.answer) for r, o in ((r, ask(ns, r.question, None)) for r in rows)])

    print(f"student  (no ref)   mean {s.mean():.3f}  median {np.median(s):.3f}", flush=True)
    MR_release = ns.get("release")

    if MR_release:
        MR_release("student", "tok")

    gc.collect(); mx.clear_cache()

    # cc[3] and cc[4] used to be these two; they are now the memory helper and the
    # MedQuAD loader, and the notebook they were read from no longer exists
    cc = nb_cells()

    exec(cell(cc, "teacher, teacher_tok ="), ns)   # loads the teacher
    exec(cell(cc, "def _mk("), ns)                 # prompt templates and clean_teacher_text

    smp = make_sampler(temp=0.3, top_p=0.9)
    clean = ns["clean_teacher_text"]

    # batch the teacher in eights, which is what fits beside the student
    def run(prompts):
        out = []

        for i in range(0, len(prompts), 8):
            out += batch_generate(ns["teacher"], ns["teacher_tok"], prompts[i:i+8], max_tokens=380, sampler=smp, verbose=False).texts

        return out
    
    p_un = [ns["_mk"](f"Question: {r.question}\n\nAnswer in 3-5 clear sentences.") for r in rows]
    
    t_un = np.array([ground(clean(t), r.answer) for t, r in zip(run(p_un), rows)])
    print(f"teacher  (no ref)   mean {t_un.mean():.3f}  median {np.median(t_un):.3f}")

    p_ref = [ns["_mk"](ns["_user_answer"](r.question, r.answer)) for r in rows]
    t_ref = np.array([ground(clean(t), r.answer) for t, r in zip(run(p_ref), rows)])

    print(f"teacher  (WITH ref) mean {t_ref.mean():.3f}  median {np.median(t_ref):.3f}")

    for label, arr in (("teacher_noref - student", t_un), ("teacher_ref - student", t_ref)):
        d = arr - s
        lo, hi = boot_ci(d)

        print(f"{label}: {d.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"{'SIGNIFICANT' if (lo > 0 or hi < 0) else 'not significant'}")


# base against shipped adapter on identical references, so only weights differ
def cmd_ablate(args):
    ns, rows = setup(args.n)
    ix = MR.get_index(MR.DEFAULT_CSV, MR.INDEX_CACHE)

    print(f"\n  same {len(rows)} held-out questions, same retrieved reference, greedy\n")

    # score one set of weights over every held-out question
    def arm(label, adapter):
        
        if adapter:
            load_adapter(ns, adapter)

        g, tags, lens = [], 0, []

        for i, row in enumerate(rows):
            msgs, _, _, _ = MR.build_messages(row.question, ix, 2.0, 2600, False)
            user = msgs[-1]["content"] if msgs else row.question
            out = ask(ns, user, None, max_tokens=420)

            m = TAGGED.search(out)
            a = (m.group(2) if m else out).strip()

            g.append(ground(a, row.answer)); tags += bool(m); lens.append(len(a))
            print(f"    {label} {i+1}/{len(rows)}", end="\r")

        n = len(rows)

        print(f"  {label:<22} grounding {np.mean(g):.3f}   format {100*tags/n:.1f}%   " f"length {int(np.mean(lens))} chars")
        
        return np.mean(g), tags / n

    b = arm("BASE Llama-3.2-1B", None)
    s = arm("SFT (2500 steps)", args.adapter)

    print(f"\n  delta: grounding {s[0]-b[0]:+.3f}   format {100*(s[1]-b[1]):+.1f} points")

# subcommands: compare, refusal, ceiling, ablate
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare"); c.set_defaults(fn=cmd_compare)
    
    c.add_argument("--adapters", nargs="+", default=["sft_adapter"])
    c.add_argument("--no-rag", action="store_true")
    c.add_argument("-n", type=int, default=60)

    r = sub.add_parser("refusal"); r.set_defaults(fn=cmd_refusal)
    r.add_argument("--adapter", default="sft_adapter")
    r.add_argument("--strict", action="store_true", help="judge retrieval by exact row instead of same-condition+facet")
    r.add_argument("-n", type=int, default=60)

    e = sub.add_parser("ceiling"); e.set_defaults(fn=cmd_ceiling)
    e.add_argument("--adapter", default="sft_adapter")
    e.add_argument("-n", type=int, default=60)

    a = sub.add_parser("ablate"); a.set_defaults(fn=cmd_ablate)
    a.add_argument("--adapter", default="sft_adapter")
    a.add_argument("-n", type=int, default=40)

    args = ap.parse_args()
    args.fn(args)

if __name__ == "__main__":
    main()