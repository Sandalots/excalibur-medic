#!/usr/bin/env python3
"""Model-free regression suite for retrieval, guards and the verbatim path.
Run after ANY change to the corpus, the alias map, or excalibur_inference.py:
Needs no model and no network: pure BM25 over the CSVs, a few seconds. """
import csv, importlib, pathlib, random, re, resource, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import excalibur_inference as M
# drop the derived corpus and index, then reload -- otherwise this checks
# whatever was cached last, not what the sources say now
pathlib.Path(M._COMBINED).unlink(missing_ok=True)
pathlib.Path(M.INDEX_CACHE).unlink(missing_ok=True)
importlib.reload(M)

# read the drug corpus straight from the csv
rows = list(csv.DictReader(open(M._DRUGS, newline="", encoding="utf-8")))

print(f"drug corpus : {len(rows)} docs, {len({r['focus_area'] for r in rows})} drugs, "
      f"{sum(len(r['answer']) for r in rows)/1024:.0f} KB")

# sections stopped at the cap, and sections that end mid-sentence
cap = [r for r in rows if len(r["answer"]) >= 8990]
mid = [r for r in rows if r["answer"].endswith(("...", "…")) or (len(r["answer"]) > 200 and not r["answer"].rstrip().endswith((".", ")", "%", ":")))]

print(f"\n1. TRUNCATION")
print(f"   at the 9000 cap        : {len(cap)}/{len(rows)}")
print(f"   not ending cleanly     : {len(mid)}/{len(rows)}")
print(f"   longest doc            : {max(len(r['answer']) for r in rows)} chars")

print(f"\n2. BRAND NAMES  (aliases loaded: {len(M.ALIASES)})")

# build the index once and time it; every check below reuses it
t = time.time(); ix = M.BM25Index.build(str(M.DEFAULT_CSV)); build = time.time() - t

# brands people actually say, each with the generic it must reach
brands = [("What is Lipitor?", "atorvastatin"), ("What is Glucophage?", "metformin"), ("What is Synthroid?", "levothyroxine"), ("What is Zoloft?", "sertraline"), ("What is Tylenol?", "acetaminophen"), ("What is Advil?", "ibuprofen"), ("What is Ventolin?", "albuterol"), ("What is Prilosec?", "omeprazole"), ("What is the dose of Lipitor?", "atorvastatin")]

hit = 0

for q, gen in brands:
    m, g, opts, vb = M.build_messages(q, ix, 2.0, 2600, False)

    body = (vb or str(m)).lower()
    ok = (g or vb) and gen in body

    hit += bool(ok)

    print(f"   {'ok  ' if ok else 'MISS'} {q:<30} -> {gen}")

print(f"   {hit}/{len(brands)}")

print(f"\n3. INDEX COST")
print(f"   docs {len(ix.dcond)}, build {build:.1f}s, " f"RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.0f} MB, ,"f" avglen {ix.avglen:.0f} tokens")

# a fixed sample, so the number is comparable between runs
random.seed(0)
idxs = random.sample(range(len(ix.dcond)), 1200)

strict = func = 0
t = time.time()

# does every document retrieve itself, and if not, something equivalent
for i in idxs:
    h = ix.search(M.expand_aliases(ix.docs[i][0]), k=1)

    if not h:
        continue

    d = h[0][1]
    strict += (d == i)
    
    func += (ix.dcond[d] == ix.dcond[i] and ix.dfacet[d] == ix.dfacet[i])

# A comparison line must name the corpus it compares against: a MedQuAD-only baseline
# printed beside a combined-corpus measurement invents a regression that does not exist.
print(f"\n4. RETRIEVAL (1200 held-out, combined corpus; expect strict ~82.9% / " f"functional ~97.1%)")

print(f"   strict {100*strict/1200:.1f}%   functional {100*func/1200:.1f}%   " f"{(time.time()-t)/1200*1000:.1f} ms/q")

# one case per outcome: answer, verbatim quote, refusal, disambiguation
cases = [("What are the symptoms of hypothyroidism?", "cond"), ("What is diabetes?", "cond"),
         ("What causes anemia?", "cond"), ("How to treat asthma?", "cond"),
         ("What is dialysis?", "cond"), ("What are the symptoms of cancer?", "cond"),
         ("What is the dose of metformin?", "VERB"), ("Who should not take warfarin?", "VERB"),
         ("What are the side effects of atorvastatin?", "VERB"),
         ("What interacts with sertraline?", "VERB"),
         ("What are the serious warnings for metformin?", "VERB"),
         ("What is atorvastatin?", "drug"), ("What is levothyroxine?", "drug"),
         ("What torque for a bicycle bolt?", "gate"), ("what cause guat?", "gate"),
         ("What causes kidney stones?", "disambig")]

ok = 0
print(f"\n5. BEHAVIOUR")

# each kind has its own definition of a correct outcome
for q, kind in cases:
    m, g, opts, vb = M.build_messages(q, ix, 2.0, 2600, False)

    good = bool((kind == "VERB" and vb) or (kind == "disambig" and opts) or (kind in ("cond", "drug") and g and not vb and not opts) or (kind == "gate" and not g))
    
    ok += good

    if not good:
        print(f"   BAD {q} -> vb={bool(vb)} g={g} opts={len(opts)}")

print(f"   {ok}/{len(cases)}")

# the dose quote must still carry its contraindication
m, g, opts, vb = M.build_messages("What is the dose of metformin?", ix, 2.0, 2600, False)
print(f"\n6. SAFETY CONTENT in the metformin dose quote ({len(vb or '')} chars)")

missing = [n for n in ["eGFR", "2000", "2550", "with meals"] if n.lower() not in (vb or "").lower()]

for n in ["eGFR", "2000", "2550", "with meals"]:
    print(f"   {'present' if n not in missing else 'ABSENT '}  {n}")

print(f"\n7. ALIASES MUST NOT SHADOW AN INDEXED TERM")

# an alias must never shadow something the corpus answers directly
indexed = {c for c in ix.dcond if c}
shadowing = sorted(k for k in M.ALIASES if k in indexed)

print(f"   {len(shadowing)} alias(es) name something the corpus answers directly"
      f"{': ' + ', '.join(shadowing[:5]) if shadowing else ''}")

# and must steer each question to the drug that was asked about
steered = []

for q, want in [("What is the dose of omeprazole?", "omeprazole"), ("What is the dose of esomeprazole?", "esomeprazole"), ("What is mupirocin?", "mupirocin"), ("What is the dose of metformin?", "metformin"), ("What is the dose of Prilosec?", "omeprazole"), ("What is Lipitor?", "atorvastatin")]:
    hit = want in M.expand_aliases(q).lower()

    print(f"   {'ok  ' if hit else 'WRONG'} {q:36s} -> {M.expand_aliases(q)[:52]}")

    if not hit:
        steered.append(q)

print(f"\n8. POSSESSIVES SURVIVE A MISSING APOSTROPHE")
# every condition the corpus writes with an apostrophe
poss_names = sorted({(r.get("focus_area") or "").strip() for r in csv.DictReader(open(M.DEFAULT_CSV, newline="", encoding="utf-8")) if re.search(r"\b[A-Za-z]{3,}'s\b", r.get("focus_area") or "")})

with_ap = without = 0

# ask each one both ways -- users do not type the apostrophe
for n in poss_names:
    
    for q, is_ap in ((f"What is {n}?", True), (f"What is {n.replace(chr(39) + 's', 's')}?", False)):
        _, g, o, _ = M.build_messages(q, ix, 2.0, 2600, False)

        if g and not o:
            with_ap += is_ap
            without += not is_ap

print(f"   {len(poss_names)} possessive condition names")
print(f"   with apostrophe    {with_ap}/{len(poss_names)}")
print(f"   without apostrophe {without}/{len(poss_names)}   (16 before the fix)")

# any threshold below fails the whole suite
poss_fail = without < len(poss_names) * 0.9

fail = (missing or ok < len(cases) or func / 1200 < 0.955 or strict / 1200 < 0.80 or shadowing or steered or poss_fail)

print(f"\n{'FAIL' if fail else 'PASS'}")
sys.exit(1 if fail else 0)