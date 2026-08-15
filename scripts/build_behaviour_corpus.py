#!/usr/bin/env python3
"""Behaviour training data: abstention, clarification, and de-enumerated tables.

Facts do not fit in a 1B model; behaviours do, and behaviours are what separate an
assistant from a search box. This builds the three the device was missing, to be mixed
into the CoT corpus before SFT."""
from __future__ import annotations
import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

# resolve paths against the repo root, not the caller
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import excalibur_inference as MR  # noqa: E402

OUT = ROOT / "artifacts" / "synthetic"
SEED = 1337

# (question, how to name the topic in the refusal). plausible-sounding rather than
# absurd -- the model already declines "what is 2+2"; it fails on things that sound
# clinical. two kinds of refusal: a bicycle bolt does not need "someone who can
# examine you", and picking a template at random produced exactly that mismatch
OOD_TEMPLATE = (
    "<reasoning>\nThis is not a medical question. It concerns {topic}, and nothing in a "
    "corpus of conditions, medicines and health topics addresses it. Answering from memory "
    "would mean inventing specifics that sound authoritative.\n</reasoning>\n"
    "<answer>\nThat is outside what I cover. My references are medical — conditions, "
    "medicines and health topics from NIH and FDA sources — so I have nothing to answer "
    "this from, and I will not guess at numbers.\n</answer>")

PERSONAL_TEMPLATE = (
    "<reasoning>\nThis is a medical question, but it is about THIS person specifically. "
    "Answering it needs their history, examination and results, none of which I have. A "
    "general reference cannot substitute for that.\n</reasoning>\n"
    "<answer>\nI cannot answer that one. It depends on your own history, examination and "
    "results, which I have no way of seeing. Please ask the clinician looking after you — "
    "and if this is urgent, seek care now rather than waiting.\n</answer>")

CLARIFY_TEMPLATE = (
    "<reasoning>\nThe question asks about {broad}, but that covers several distinct "
    "conditions with different {facet_word}. Answering about one of them would silently "
    "substitute a narrower question for the one asked.\n</reasoning>\n"
    "<answer>\nI have references for several specific forms rather than {broad} as a "
    "whole. Which did you mean?\n{options}\n</answer>")

FACET_WORD = {"treat": "treatments", "symptom": "symptoms", "cause": "causes", "diagnos": "diagnostic approaches", "prevent": "prevention", "risk": "risk factors", "define": "definitions"}

def load_corpus():
    path = MR.DEFAULT_CSV if MR.DEFAULT_CSV.exists() else MR._MEDQUAD

    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("answer") and r.get("question")]

# Phrasings the same refusal has to survive. Twelve fixed strings would be memorised;
# the behaviour has to generalise to wording the model has not seen.
ASK = ["What is the {x}?", "How much {x} is correct?", "Can you tell me the {x}?", "I need to know the {x}.", "What would you recommend for the {x}?", "Quick question — the {x}?"]

OOD_SUBJECTS = [
    ("torque for a bicycle crank bolt", "bicycle maintenance"),
    ("tyre pressure for a road bike", "bicycle maintenance"),
    ("fertiliser rate for tomatoes", "horticulture"),
    ("chlorine dose for a swimming pool", "pool chemistry"),
    ("current rating of a domestic ring main", "domestic electrical work"),
    ("safe working load of a 10 mm steel cable", "mechanical engineering"),
    ("oil viscosity for a diesel engine", "engine maintenance"),
    ("curing time for concrete before loading", "construction"),
    ("caffeine limit for a racehorse", "veterinary and equine dosing"),
    ("antibiotic withdrawal period in dairy cattle", "veterinary medicine"),
    ("LD50 of paracetamol in mice", "laboratory toxicology in animals"),
    ("antibiotic for kennel cough in a dog", "veterinary medicine"),
    ("worming dose for a cat", "veterinary medicine"),
    ("nitrogen level for a lawn", "horticulture"),
    ("water hardness for a fish tank", "aquarium keeping"),
]

PERSONAL_SUBJECTS = [
    "What is the prognosis for stage 4 pancreatic cancer in a 78-year-old?",
    "Should I stop my anticoagulant before dental extraction?",
    "What does a troponin of 0.9 mean in my case?",
    "Is my rash caused by the new medication I started last week?",
    "How long will my recovery take after this surgery?",
    "Which of these two treatments is better for me?",
    "What did my scan results mean?",
    "Should I increase my dose if it is not working?",
    "Is this chest pain something I should worry about?",
    "Can I drink alcohol with the tablets I am on?",
    "My child has a fever of 39 — should I go to hospital?",
    "Do these symptoms mean I have cancer?",
    "Is it safe for me to stop taking this?",
    "Why did my doctor prescribe this instead of the other one?",
    "How long do I have?",
]

def build_abstention(rng) -> list[dict]:
    rows = []

    # four phrasings each, so the behaviour survives wording it has not seen
    for subject, topic in OOD_SUBJECTS:

        for tmpl in rng.sample(ASK, 4):
            rows.append({"question": tmpl.format(x=subject), "reference": "", "focus_area": "__abstain_ood__", "split": "train", "completion": OOD_TEMPLATE.format(topic=topic)})
            
    # personal questions get their own template and three lead-ins
    for q in PERSONAL_SUBJECTS:
        rows.append({"question": q, "reference": "", "focus_area": "__abstain_personal__", "split": "train", "completion": PERSONAL_TEMPLATE})
        
        for lead in ("Please just tell me — ", "I know you are not a doctor, but ", "Roughly, "):
            
            rows.append({"question": lead + q[0].lower() + q[1:], "reference": "", "focus_area": "__abstain_personal__", "split": "train", "completion": PERSONAL_TEMPLATE})
    return rows

def build_clarification(corpus, rng, limit=40) -> list[dict]:
    conds = {}

    # index every condition by the facets the corpus actually holds for it
    for r in corpus:
        c = MR.condition_of(r["question"])

        if c:
            conds.setdefault(c, set()).add(MR.facet_of(r["question"]))

    # a condition is broad if at least three narrower ones sit under it
    broad = [c for c in conds if sum(1 for o in conds if o != c and o.startswith(c + " ")) >= 3]

    rng.shuffle(broad)

    rows = []

    # offer real narrower options, never invented ones
    for b in broad[:limit]:
        narrow = sorted(o for o in conds if o != b and o.startswith(b + " "))[:5]

        if len(narrow) < 3:
            continue

        facet = rng.choice(["treat", "symptom", "cause"])
        qtmpl = {"treat": "How do you treat {}?", "symptom": "What are the symptoms of {}?", "cause": "What causes {}?"}[facet]
        
        rows.append({"question": qtmpl.format(b), "reference": "", "focus_area": "__clarify__", "split": "train", "completion": CLARIFY_TEMPLATE.format(broad=b, facet_word=FACET_WORD[facet], options="\n".join(f"  - {n}" for n in narrow))})
        
    return rows

HPO_RE = re.compile(r"Human Phenotype Ontology", re.I)
FREQ_RE = re.compile(r"([A-Z][A-Za-z /'\-]{3,40}?)\s+(\d+(?:\.\d+)?)%")

def deenumerate(reference: str) -> str | None:
    pairs = [(m.group(1).strip(), float(m.group(2))) for m in FREQ_RE.finditer(reference)]

    if len(pairs) < 4:
        # many HPO tables list signs with no frequency at all. no numeric risk, but
        # the same list-dumping, and 46% of the HPO rows. summarise them the same way
        bare = [n.strip() for n in re.findall(r"([A-Z][A-Za-z /'\-]{3,40}?)\s+-\s", reference)]
        bare = [b for b in bare if len(b.split()) <= 5][:6]

        if len(bare) < 4:
            return None
        
        listed = ", ".join(b.lower() for b in bare[:4])

        return (f"<reasoning>\nThe reference lists the reported signs without frequencies, "
                f"so no one of them can be called typical. Naming several and saying the "
                f"list is longer is more honest than reciting all of it as though it were "
                f"a description of one patient.\n</reasoning>\n"
                f"<answer>\nReported signs include {listed}, among others. The reference "
                f"lists them without frequencies, so how often each occurs is not stated — "
                f"an individual would not be expected to have all of them.\n</answer>")

    pairs.sort(key=lambda p: -p[1])
    top = pairs[:4]
    rest = len(pairs) - len(top)
    listed = ", ".join(f"{n.lower()} ({f:g}%)" for n, f in top)

    return (f"<reasoning>\nThe reference is a frequency table: each sign carries its own "
            f"percentage, and they differ widely — from {top[0][1]:g}% down to "
            f"{pairs[-1][1]:g}%. Naming the most common ones with their own figures is "
            f"more useful than reciting the whole list, and avoids implying that one "
            f"frequency applies to all of them.\n</reasoning>\n"
            f"<answer>\nThe most frequently reported signs are {listed}. "
            f"{'A further ' + str(rest) + ' signs are listed at lower or unstated frequencies. ' if rest else ''}"
            f"Frequencies are per sign, not for the condition as a whole, and come from "
            f"studies of small numbers of patients.\n</answer>")

def build_deenumerated(corpus, rng, limit=None) -> list[dict]:
    sft = OUT / "sft_train.jsonl"

    if sft.exists():
        source = [json.loads(l) for l in open(sft)]
        source = [r for r in source if HPO_RE.search(r.get("reference", ""))]
        
        print(f"  deenum source: {len(source)} HPO rows in sft_train.jsonl")

    else:
        source = [{"question": r["question"], "reference": r["answer"]} for r in corpus if HPO_RE.search(r["answer"])]
        print(f"  deenum source: sft_train.jsonl absent, using {len(source)} corpus rows")
        
        rng.shuffle(source)

    rows, skipped = [], 0

    for r in source:
        c = deenumerate(r["reference"])

        if not c:
            skipped += 1

            continue

        rows.append({"question": r["question"], "reference": r["reference"], "focus_area": r.get("focus_area") or MR.condition_of(r["question"]) or "unknown", "split": r.get("split", "train"), "completion": c})
        
        if limit and len(rows) >= limit:
            break

    if skipped:
        print(f"  deenum: {skipped} rows had fewer than 4 parseable frequencies — " f"left as they are")
        
    return rows

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="print samples, write nothing")

    args = ap.parse_args()

    rng = random.Random(SEED)

    # the same corpus the runner indexes
    corpus = load_corpus()
    print(f"  corpus: {len(corpus):,} rows")

    # build the three behaviours
    parts = {
        "abstain": build_abstention(rng),
        "clarify": build_clarification(corpus, rng),
        "deenum": build_deenumerated(corpus, rng),
    }

    for k, v in parts.items():
        print(f"  {k:<9} {len(v):>4} examples")

    # preview prints one sample of each and writes nothing
    if args.preview:
        for k, v in parts.items():

            if not v:
                continue

            print(f"\n{'='*70}\n  {k.upper()}\n{'='*70}")
            
            s = v[0]

            print(f"  Q: {s['question']}")
            print("  " + s["completion"][:520].replace("\n", "\n  "))

        return

    # write both files: added examples, and replacements keyed by question
    OUT.mkdir(parents=True, exist_ok=True)

    # added versus REPLACED. de-enumerated examples must displace the HPO samples,
    # keyed by question -- adding them alongside teaches both behaviours at once
    added = parts["abstain"] + parts["clarify"]
    replacements = {r["question"]: r for r in parts["deenum"]}

    ap_path = OUT / "behaviour_added.jsonl"

    with open(ap_path, "w") as f:

        for r in added:
            f.write(json.dumps(r) + "\n")

    rp_path = OUT / "behaviour_replacements.jsonl"

    with open(rp_path, "w") as f:
        
        for r in replacements.values():
            f.write(json.dumps(r) + "\n")

    print(f"\n  wrote {ap_path.relative_to(ROOT)}: {len(added)} NEW examples")
    print(f"  wrote {rp_path.relative_to(ROOT)}: {len(replacements)} REPLACEMENTS " f"(match on `question`)")

    # how the mix lands, against the measured 4,900-sample train set
    sft = OUT / "sft_train.jsonl"

    if sft.exists():
        existing = [json.loads(l) for l in open(sft)]
        hpo = sum(1 for r in existing if "Human Phenotype Ontology" in r.get("reference", ""))
        matched = sum(1 for r in existing if r["question"] in replacements)
        
        total = len(existing) + len(added)

        print(f"\n  against sft_train.jsonl ({len(existing):,} samples):")
        print(f"    HPO-derived samples present   {hpo:>5}")
        print(f"    of those, replaceable here    {matched:>5}")
        print(f"    combined training set         {total:>5,}")
        print(f"    abstention share              {100*len(parts['abstain'])/total:>5.1f}%")
        print(f"    clarification share           {100*len(parts['clarify'])/total:>5.1f}%")

        if matched < hpo * 0.5:
            print(f"\n    NOTE: only {100*matched/max(hpo,1):.0f}% of the HPO samples have a replacement.\n    Raise --limit, or drop the unmatched ones instead.")

if __name__ == "__main__":
    main()