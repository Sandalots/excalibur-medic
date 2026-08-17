#!/usr/bin/env python3
"""Build the retrieval corpus from two public-domain sources.
Both emit the SAME four-column MedQuAD-style row, question, answer, source, focus_area, and that is the whole trick: `condition_of()` and `facet_of()` parse the question text,
so a drug label and a health topic reach retrieval through the code MedQuAD already uses.
Neither source needed retrieval code of its own."""
from __future__ import annotations
import argparse
import collections
import csv
import html
import importlib.util
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
MEDQUAD = ROOT / "datasets" / "medquad.csv"
UA = {"User-Agent": "EXCALIBUR-research/1.0 (offline medical reference)"}

# store sections long, trim at use -- a 2400-char cap cut 176 safety-critical
# sections mid-sentence, and verbatim quotes never reach the model anyway
MAX_CHARS = 9000
MIN_CHARS_DRUG, MIN_CHARS_TOPIC = 150, 120
ROW_FIELDS = ["question", "answer", "source", "focus_area"]

# shared utility functions for both fetchers
def http_json(url: str, timeout: int = 30):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return json.load(r)

# GET a URL and decode it, replacing what will not decode
def http_text(url: str, timeout: int = 30) -> str:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")

# trim to a length without cutting a word in half
def cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    
    head = text[:limit]

    for sep in (". ", "; ", ", "):
        i = head.rfind(sep)

        if i > limit * 0.6:
            return head[:i + 1].rstrip()
        
    return head.rsplit(" ", 1)[0].rstrip() + " ..."

# write a corpus CSV, making the directory if needed
def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS)

        w.writeheader()
        w.writerows(rows)

# sorted JSON, so a re-fetch produces a readable diff
def write_aliases(path: Path, mapping: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=0, sort_keys=True))

# every focus_area written so far, as raw, letters-only and single words
def focus_areas(*csvs: Path) -> tuple[set[str], set[str], set[str]]:
    verbatim: set[str] = set()
    letters: set[str] = set()
    words: set[str] = set()

    for c in csvs:

        if not c.exists():
            continue

        with open(c, newline="", encoding="utf-8") as f:

            for row in csv.DictReader(f):
                fa = (row.get("focus_area") or "").strip().lower()

                if not fa:
                    continue

                verbatim.add(fa)
                s = re.sub(r"[^a-z ]", " ", fa).strip()

                if s:
                    letters.add(s)
                    words.update(s.split())

    return verbatim, letters, words

# import excalibur_inference by path, to reuse its facet rules
def runner():
    spec = importlib.util.spec_from_file_location("_mr", ROOT / "excalibur_inference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod

# drop synonyms carrying facet words -- they misroute a dose question
def drop_facet_aliases(aliases: dict[str, str]) -> dict[str, str]:
    mr = runner()

    bad = {k: v for k, v in aliases.items() if mr.facet_of(k) != "other" or mr.facet_of(v) != "other"}
    
    if bad:
        print(f"dropped {len(bad)} synonyms carrying facet vocabulary " f"(e.g. {sorted(bad)[:5]})")
        
    return {k: v for k, v in aliases.items() if k not in bad}

# openFDA drug labels
# query by generic name and reject combinations -- a naive "metformin" search
# returns ZITUVIMET (sitagliptin + metformin)

DRUG_CSV = ART / "drug_labels.csv"
DRUG_ALIASES = ART / "drug_aliases.json"
TOPIC_CSV = ART / "medlineplus.csv"
TOPIC_ALIASES = ART / "topic_aliases.json"

WS = "https://wsearch.nlm.nih.gov/ws/query"
PUBLIC_DOMAIN_ORG = "National Library of Medicine"
RETMAX = 8
PAUSE = 0.34 # NLM asks for <= ~3 requests/second

DRUGS = """
metformin glipizide glyburide glimepiride sitagliptin empagliflozin dapagliflozin
pioglitazone insulin acarbose
atorvastatin simvastatin rosuvastatin pravastatin lovastatin ezetimibe fenofibrate
gemfibrozil niacin
lisinopril enalapril ramipril losartan valsartan irbesartan olmesartan candesartan
amlodipine nifedipine diltiazem verapamil metoprolol atenolol carvedilol bisoprolol
propranolol nebivolol labetalol
hydrochlorothiazide chlorthalidone furosemide bumetanide spironolactone triamterene
clonidine hydralazine isosorbide doxazosin terazosin
warfarin clopidogrel apixaban rivaroxaban dabigatran ticagrelor prasugrel dipyridamole
cilostazol enoxaparin
omeprazole pantoprazole esomeprazole lansoprazole famotidine ranitidine sucralfate
ondansetron metoclopramide promethazine loperamide docusate bisacodyl polyethylene
mesalamine
levothyroxine liothyronine methimazole propylthiouracil prednisone prednisolone
methylprednisolone hydrocortisone dexamethasone fludrocortisone
alendronate risedronate ibandronate raloxifene calcitriol cholecalciferol
sertraline fluoxetine citalopram escitalopram paroxetine venlafaxine duloxetine
bupropion mirtazapine trazodone amitriptyline nortriptyline doxepin
alprazolam lorazepam clonazepam diazepam temazepam zolpidem eszopiclone buspirone
hydroxyzine
quetiapine risperidone olanzapine aripiprazole ziprasidone haloperidol lithium
lamotrigine valproate carbamazepine oxcarbazepine levetiracetam phenytoin topiramate
gabapentin pregabalin zonisamide primidone
donepezil memantine rivastigmine galantamine levodopa carbidopa pramipexole ropinirole
amantadine benztropine
sumatriptan rizatriptan zolmitriptan propranolol topiramate
ibuprofen naproxen diclofenac meloxicam celecoxib indomethacin ketorolac
acetaminophen aspirin tramadol codeine morphine oxycodone hydrocodone hydromorphone
fentanyl methadone buprenorphine naloxone naltrexone
cyclobenzaprine baclofen tizanidine methocarbamol carisoprodol
amoxicillin penicillin ampicillin cephalexin cefdinir cefuroxime ceftriaxone
azithromycin clarithromycin erythromycin doxycycline minocycline tetracycline
ciprofloxacin levofloxacin moxifloxacin trimethoprim sulfamethoxazole nitrofurantoin
metronidazole clindamycin vancomycin linezolid rifampin isoniazid ethambutol
pyrazinamide
fluconazole itraconazole terbinafine nystatin ketoconazole griseofulvin
acyclovir valacyclovir oseltamivir famciclovir ribavirin
hydroxychloroquine methotrexate sulfasalazine leflunomide azathioprine mycophenolate
tacrolimus cyclosporine allopurinol colchicine probenecid febuxostat
albuterol ipratropium tiotropium salmeterol formoterol budesonide fluticasone
montelukast theophylline prednisone azelastine
loratadine cetirizine fexofenadine diphenhydramine chlorpheniramine pseudoephedrine
guaifenesin dextromethorphan benzonatate
tamsulosin finasteride dutasteride sildenafil tadalafil oxybutynin tolterodine
solifenacin mirabegron
estradiol progesterone medroxyprogesterone norethindrone levonorgestrel
testosterone clomiphene letrozole anastrozole tamoxifen
ferrous folic cyanocobalamin thiamine pyridoxine ascorbic magnesium potassium
calcium melatonin
latanoprost timolol brimonidine dorzolamide erythromycin
hydrocortisone triamcinolone clobetasol mupirocin ketoconazole permethrin
tretinoin clindamycin benzoyl adapalene isotretinoin
""".split()

# label section -> (MedQuAD-style question template, facet it should resolve to)
SECTIONS = [
    ("indications_and_usage",      "What is (are) {d} ?",                  "define"),
    ("dosage_and_administration",  "What is the dose of {d} ?",            "dose"),
    ("adverse_reactions",          "What are the side effects of {d} ?",   "sideeffect"),
    ("drug_interactions",          "What interacts with {d} ?",            "interact"),
    ("contraindications",          "Who should not take {d} ?",            "contra"),
    ("boxed_warning",              "What are the serious warnings for {d} ?", "warning"),
    ("warnings_and_cautions",      "What precautions apply to {d} ?",      "warning"),
    ("use_in_specific_populations", "Is {d} safe in pregnancy ?",           "pregnancy"),
]

# otc labels use a different section vocabulary -- acetaminophen and aspirin
# yielded only 2 sections until these were added
OTC_SECTIONS = [
    ("purpose",                   "What is (are) {d} ?",                  "define"),
    ("indications_and_usage",     "What is (are) {d} ?",                  "define"),
    ("dosage_and_administration", "What is the dose of {d} ?",            "dose"),
    ("warnings",                  "What precautions apply to {d} ?",      "warning"),
    ("do_not_use",                "Who should not take {d} ?",            "contra"),
    ("stop_use",                  "When should I stop taking {d} ?",      "warning"),
    ("ask_doctor_or_pharmacist",  "What interacts with {d} ?",            "interact"),
    ("pregnancy_or_breast_feeding", "Is {d} safe in pregnancy ?",           "pregnancy"),
]

# salt/ester forms that still denote a SINGLE ingredient
SALTS = ("sodium", "hydrochloride", "hcl", "calcium", "potassium", "besylate",
         "tartrate", "sulfate", "succinate", "maleate", "citrate", "acetate",
         "phosphate", "mesylate", "fumarate", "tablets", "capsules", "usp",
         "medoxomil", "etexilate", "mononitrate", "dinitrate", "bitartrate",
         "hydrobromide", "sulphate", "dipropionate", "propionate", "valerate",
         "furoate", "xinafoate", "bromide", "chloride", "nitrate", "oxide",
         "monohydrate", "anhydrous", "micronized", "extended", "release",
         "delayed", "er", "xr", "sr", "dr")

# must be a HEADING, not the word in passing. A bare \bPregnancy\b match started
# glyburide's answer mid-sentence at "...pregnancy or for use in pediatric patients."
PREG_START = re.compile(r"8\.\d+\s+Pregnancy\b|\bPregnancy\s*(?:Risk Summary|:)", re.I)
PREG_END = re.compile(r"8\.\d+\s+(?:Pediatric|Geriatric|Renal|Hepatic|Females and Males)"
                      r"|(?<![\w.])(?:Pediatric Use|Geriatric Use|Renal Impairment|"
                      r"Hepatic Impairment)\b", re.I)

# keep only the pregnancy section, which openFDA files under a longer heading
def pregnancy_only(text: str) -> str:
    m = PREG_START.search(text)

    if not m:
        return ""
    
    tail = text[m.start():]
    e = PREG_END.search(tail, 1)
    out = (tail[: e.start()] if e else tail).strip()

    return out if len(out) >= MIN_CHARS_DRUG else ""

# normalise label text and strip the FDA cross-references read as incidence
def clean(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    # strip fda self-references like "( 6.1 )" -- a model read them as incidence
    # rates and emitted five fabricated percentages, all reported as grounded
    t = re.sub(r"\[\s*see\b[^\]]{0,120}\]", " ", t, flags=re.I)
    t = re.sub(r"\(\s*\d+\.\d+\s*\)", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    # labels start with their own section numbering: "1 INDICATIONS AND USAGE ..."
    t = re.sub(r"^\d+(\.\d+)?\s+[A-Z][A-Z \-/,&]{4,}\s+", "", t)

    return t

# one openFDA label for a generic, or None when nothing usable returns
def fetch_label(generic: str) -> dict | None:
    q = urllib.parse.quote(f'openfda.generic_name:"{generic}"')

    url = f"https://api.fda.gov/drug/label.json?search={q}&limit=25"

    try:
        results = http_json(url).get("results", [])

    except Exception as e:
        print(f"  {generic:<22} FETCH FAILED: {str(e)[:60]}")

        return None
    
    best, best_score = None, -1

    for res in results:
        names = [n.lower() for n in (res.get("openfda", {}).get("generic_name") or [])]

        if not names:
            continue

        name = names[0]

        if generic not in name:
            continue

        # reject combinations: " and " missed hyphenated pairs like
        # OLMESARTAN-HYDROCHLOROTHIAZIDE. strip salts, then require an exact match
        stem = re.sub(r"[^a-z ]", " ", name)

        for salt in SALTS:
            stem = re.sub(rf"\b{salt}\b", " ", stem)
            
        stem = " ".join(stem.split())

        if stem != generic:
            continue

        score = sum(1 for k, _, _ in SECTIONS + OTC_SECTIONS if res.get(k))

        if score > best_score:
            best, best_score = res, score

    return best

# reject a label whose generic name does not match what was asked for
def validate(res: dict, generic: str) -> str | None:
    ofda = res.get("openfda", {})
    names = [n.lower() for n in (ofda.get("generic_name") or [])]

    if not names:
        return "no generic_name"
    
    if len(names) > 1:
        return f"multi-ingredient ({len(names)} generics)"
    
    # do not test spl_product_data_elements for length -- it lists inactive
    # ingredients, and a length rule here once rejected 175 of 285 drugs
    n_sec = sum(1 for k, _, _ in SECTIONS + OTC_SECTIONS if res.get(k))

    if n_sec < 2:
        return f"only {n_sec} usable section(s)"
    
    return None

# map brand names onto indexed generics, so a trade name still retrieves
def brand_aliases(indexed: set[str]) -> dict[str, str]:
    seen: dict[str, set[str]] = {}

    for d in DRUGS:
        q = urllib.parse.quote(f'openfda.generic_name:"{d}"')

        url = (f"https://api.fda.gov/drug/label.json?search={q}" f"&count=openfda.brand_name.exact&limit=100")

        try:
            terms = http_json(url).get("results", [])

        except Exception:
            continue

        for t in terms:
            b = (t.get("term") or "").lower().strip()
            # no minimum count: a >=2 floor removed every brand people name
            # pfizer publishes one Lipitor label. measured 0/9 with it in place

            if not b or len(b) > 28:
                continue

            if d in b or " and " in b or "-" in b or "/" in b or "+" in b:
                continue

            if not re.fullmatch(r"[a-z][a-z0-9 ']{2,}", b):
                continue

            seen.setdefault(b, set()).add(d)

        time.sleep(0.2)

    # a brand naming two different generics is ambiguous -- drop it rather than answer
    # about the wrong drug. (An earlier setdefault silently kept whichever came first.)
    out = {b: next(iter(g)) for b, g in seen.items() if len(g) == 1}

    # openFDA stores decorated brands ("VENTOLIN HFA"), never bare "TYLENOL", so
    # derive single words -- but only where every alias using one points at the
    # same generic. the >=6 rule is load-bearing: at >=4 it derived "dose" ->
    # levonorgestrel and retrieval fell 90.0 -> 88.2% strict
    owners: dict[str, set[str]] = {}

    for b, g in out.items():
        for w in b.split():
            
            if len(w) >= 6:
                owners.setdefault(w, set()).add(g)

    for w, gens in owners.items():

        if len(gens) == 1 and w not in out and w not in DRUGS:
            out[w] = next(iter(gens))

    # critical. otc brand names are ordinary medical words -- ALLERGY, SLEEP AID --
    # and aliasing them rewrites "what is allergy?" into a drug question. a term
    # naming a condition in MedQuAD is never a brand
    #
    # an alias must point AT a drug we hold, never AWAY from one: omeprazole was
    # aliased to its salt counter-ion and its dose question quoted the MAGNESIUM
    # label. prune_aliases() checks again at load, since the file ships as data
    orphan = {b for b, g in out.items() if g not in indexed}
    shadow = {b for b in out if b in indexed}

    if orphan or shadow:
        print(f"  dropped {len(orphan)} aliases pointing at an unindexed generic, "
              f"{len(shadow)} that name an indexed drug themselves "
              f"{sorted(shadow)[:4] if shadow else ''}")
        
        out = {b: g for b, g in out.items() if b not in orphan | shadow}

    _, conditions, words = focus_areas(MEDQUAD)

    dropped = {b: g for b, g in out.items() if b in conditions or all(w in words for w in b.split())}
    
    out = {b: g for b, g in out.items() if b not in dropped}

    print(f"{len(out)} brand names -> generic ({len(dropped)} rejected as medical words)")

    if dropped:
        print(f"  rejected e.g. {sorted(dropped)[:8]}")
        
    return out

# fetch, validate and clean every drug label, then write the CSV and aliases
def run_drugs() -> None:
    rows = []
    rejected: list[tuple[str, str]] = []

    for d in DRUGS:
        res = fetch_label(d)

        if not res:
            rejected.append((d, "no single-ingredient label"))

            continue

        why = validate(res, d)

        if why:
            rejected.append((d, why))

            continue

        got = []

        is_otc = bool(res.get("purpose") or res.get("do_not_use"))

        for key, tmpl, facet in (OTC_SECTIONS if is_otc else SECTIONS):
            val = res.get(key)

            if not val:
                continue

            body = cut(clean(" ".join(val)), MAX_CHARS)

            if key in ("use_in_specific_populations", "pregnancy_or_breast_feeding"):
                body = pregnancy_only(body)

            if len(body) < MIN_CHARS_DRUG:
                continue

            rows.append({"question": tmpl.format(d=d.capitalize()), "answer": body, "source": "openFDA (FDA SPL, public domain)", "focus_area": d})
            
            got.append(facet)

        if not got:
            rejected.append((d, "no sections passed the length filter"))

            continue

        if len(rows) % 200 < len(got):
            print(f"  ...{len({r['focus_area'] for r in rows})} drugs, {len(rows)} docs", flush=True)

        time.sleep(0.3)

    kept = {r["focus_area"] for r in rows}

    write_aliases(DRUG_ALIASES, drop_facet_aliases(brand_aliases(kept)))
    write_rows(DRUG_CSV, rows)

    drugs = len({r["focus_area"] for r in rows})
    chars = sum(len(r["answer"]) for r in rows)

    print(f"\nwrote {DRUG_CSV.name}: {len(rows)} docs across {drugs} drugs, {chars/1024:.0f} KB")
    print(f"  avg {chars/max(len(rows),1):.0f} chars/doc")
    print(f"\nREJECTED {len(rejected)} of {len(DRUGS)} requested:")

    for reason, n in collections.Counter(r for _, r in rejected).most_common():
        ex = [d for d, r in rejected if r == reason][:6]
        print(f"  {n:>3}  {reason:<38} e.g. {', '.join(ex)}")

# MedlinePlus health topics
# the NLM summary is already a q&a, so it splits into one row per aspect rather
# than a definition blob. facet is not set here -- facet_of() derives it from the
# question text, so this corpus cannot disagree with the runner

# seed terms covering the categories the corpus-gap battery measured as weak
# deduplicated by title, so overlapping seeds cost only a request
SEEDS = [s.strip() for s in """
anxiety|depression|PTSD|bipolar disorder|schizophrenia|panic disorder|OCD|ADHD|autism
insomnia|eating disorders|substance use|alcohol|smoking cessation|suicide|grief|stress
burns|bleeding|nosebleed|choking|fractures|sprains|concussion|poisoning|wounds|first aid
frostbite|heat illness|animal bites|insect bites|CPR
MRI|CT scan|ultrasound|X-rays|mammography|colonoscopy|endoscopy|biopsy
blood tests|urinalysis|kidney tests|liver function tests|thyroid tests|blood count
cholesterol|blood sugar|electrolytes|blood pressure
immunization|childhood immunization|adult immunization|travel health|measles|mumps
appendicitis|hernia|gallstones|kidney stones|peptic ulcer|hemorrhoids
vitamin D|vitamin B12|iron|calcium|folic acid|magnesium|potassium|zinc|vitamin C
nutrition|dietary fats|carbohydrates|protein|dietary fiber|sodium|drinking water
pregnancy|prenatal care|childbirth|breastfeeding|infertility|menopause|contraception
sexually transmitted diseases|HIV|hepatitis|tuberculosis|malaria|Lyme disease
asthma|COPD|pneumonia|bronchitis|influenza|common cold|COVID-19|allergy|sinusitis
diabetes|thyroid diseases|obesity|metabolic syndrome|gout|osteoporosis|arthritis
stroke|heart attack|heart failure|arrhythmia|peripheral arterial disease
dementia|Alzheimer disease|Parkinson disease|epilepsy|migraine|multiple sclerosis
kidney disease|liver diseases|pancreatitis|celiac disease|irritable bowel syndrome
acne|eczema|psoriasis|shingles|rashes|hair loss|skin cancer|sunburn
eye care|hearing loss|dental health|foot health|sleep disorders|exercise|back pain
fever|cough|headache|dizziness|fatigue|nausea|constipation|diarrhea|pain
""".replace("\n", "|").split("|") if s.strip()]

# summary heading -> the MedQuAD question phrasing to emit. facet_of() derives the
# facet. `define` is last because its pattern is broadest, as in _FACETS
HEADINGS = [
    (r"who is at risk|how common|how many people|risk factors", "Who is at risk for {t}? ?"),
    (r"symptom|warning sign|signs of",                          "What are the symptoms of {t} ?"),
    (r"what causes|causes of",                                  "What causes {t} ?"),
    (r"diagnos|what tests|how do i know",                       "How to diagnose {t} ?"),
    (r"treatment|how is it treated|how are .* treated",         "What are the treatments for {t} ?"),
    (r"prevent|can .* be avoided",                              "How to prevent {t} ?"),
    (r"clinical trial|research",                                "what research (or clinical trials) is being done for {t}?"),
    (r"what (is|are)|what does|types of|why do i need",         "What is (are) {t} ?"),
]

# unescape HTML and drop the search-highlighting spans MedlinePlus adds
def unescape_strip(fragment: str) -> str:
    t = html.unescape(fragment)
    t = re.sub(r"</?span[^>]*>", "", t)          # <span class="qt0"> search highlighting
    t = re.sub(r"<li>", " • ", t)
    t = re.sub(r"</(p|li|ul|ol|div|h\d)>", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t)

    return t.strip()

# turn a section heading into the question form the corpus is keyed on
def question_for(heading: str, topic: str) -> str:
    low = heading.lower()

    for pat, tmpl in HEADINGS:

        if re.search(pat, low):
            return tmpl.format(t=topic)
        
    return f"What is (are) {topic} ?"

# search MedlinePlus for a topic, raw XML back, or None
def fetch_topic(term: str) -> str | None:
    q = urllib.parse.urlencode({"db": "healthTopics", "term": term, "retmax": RETMAX})

    try:
        return http_text(f"{WS}?{q}")
    
    except Exception as e:
        print(f"  {term:<28} FETCH FAILED: {str(e)[:60]}")

        return None

# walk the <document> blocks of a response one at a time
def documents(xml: str):
    for doc in re.findall(r"<document\b.*?</document>", xml, re.S):

        # one named content field out of a document block
        def field(name: str) -> str:
            m = re.search(rf'<content name="{name}">(.*?)</content>', doc, re.S)

            return unescape_strip(m.group(1)) if m else ""
        
        m = re.search(r'<content name="FullSummary">(.*?)</content>', doc, re.S)

        yield {
            "title": field("title"),
            "org": field("organizationName"),
            "summary": m.group(1) if m else "",
            "alts": [unescape_strip(a) for a in re.findall(r'<content name="altTitle">(.*?)</content>', doc, re.S)],
        }

# split a topic summary at its headings, one row per section
def split_sections(summary: str) -> list[tuple[str, str]]:
    raw = re.sub(r"</?span[^>]*>", "", html.unescape(summary))
    heads = list(re.finditer(r"([A-Z][^<>]{4,110}\?)\s*<p>", raw))

    if not heads:
        body = unescape_strip(raw)

        return [("", body)] if len(body) >= MIN_CHARS_TOPIC else []
    
    out: list[tuple[str, str]] = []
    lead = unescape_strip(raw[: heads[0].start()])

    if len(lead) >= MIN_CHARS_TOPIC:
        out.append(("", lead))

    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(raw)
        body = unescape_strip(raw[h.end(): end])

        if len(body) >= MIN_CHARS_TOPIC:
            out.append((h.group(1).strip(), body))

    return out

# fetch every health topic, split into sections, write the CSV and aliases
def run_medlineplus() -> None:
    rows: list[dict] = []
    aliases: dict[str, str] = {}
    seen_titles: set[str] = set()
    rejected: collections.Counter = collections.Counter()

    for i, term in enumerate(SEEDS, 1):
        xml = fetch_topic(term)
        
        time.sleep(PAUSE)

        if not xml:
            continue

        new_here = 0

        for d in documents(xml):
            
            if d["org"] != PUBLIC_DOMAIN_ORG:
                rejected[d["org"] or "(no organization)"] += 1

                continue

            title = d["title"].strip()

            if not title or title.lower() in seen_titles or not d["summary"]:
                continue

            secs = split_sections(d["summary"])

            if not secs:
                continue

            seen_titles.add(title.lower())
            new_here += 1

            for head, body in secs:
                rows.append({"question": question_for(head, title), "answer": cut(body, MAX_CHARS), "source": "MedlinePlus (NLM, public domain)", "focus_area": title,})

            for a in d["alts"]:
                a = a.strip().lower()
                # a synonym is only useful if it is not already the title, and only safe
                # if it maps to ONE topic -- ambiguous ones are dropped below
                if a and a != title.lower() and len(a) > 3:
                    aliases[a] = title if aliases.get(a, title) == title else ""

        print(f"  [{i:>3}/{len(SEEDS)}] {term:<30} +{new_here} topics " f"({len(seen_titles)} total, {len(rows)} rows)")

    aliases = {k: v for k, v in aliases.items() if v}

    # altTitle conflates synonyms ("Rubeola" = Measles) with narrower topics
    # ("acromegaly" -> Growth Disorders); expanding the second routes around the
    # disambiguation guard. so drop any alias already indexed in its own right
    indexed, _, _ = focus_areas(MEDQUAD, DRUG_CSV)

    for r in rows:
        indexed.add(r["focus_area"].strip().lower())

    shadowed = {k for k in aliases if k in indexed}
    aliases = {k: v for k, v in aliases.items() if k not in shadowed}

    # critical. an alias must never contain facet vocabulary: MedlinePlus lists
    # "side effects" as a synonym of "Drug Reactions", which flipped facet_of()
    # from `sideeffect` to `define` and bypassed the verbatim path entirely
    aliases = drop_facet_aliases(aliases)

    print(f"\ndropped {len(shadowed)} synonyms that shadow an indexed condition " f"(e.g. {sorted(shadowed)[:4]})")

    write_rows(TOPIC_CSV, rows)
    write_aliases(TOPIC_ALIASES, aliases)

    chars = sum(len(r["answer"]) for r in rows)
    
    print(f"\nwrote {TOPIC_CSV.name}: {len(rows)} rows across {len(seen_titles)} topics, "
          f"{chars/1024:.0f} KB (avg {chars//max(len(rows),1)} chars/row)")
    
    print(f"wrote {TOPIC_ALIASES.name}: {len(aliases)} topic synonyms")

    if rejected:
        print(f"\nnot NLM-authored, so NOT written ({sum(rejected.values())} documents):")

        for org, n in rejected.most_common(8):
            print(f"  {n:>5}  {org}")

# cli entry point for the fetcher script. The source argument is a simple switch to run one or both fetchers.
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=["drugs", "medlineplus", "all"])

    args = ap.parse_args()
    # order matters for "all": the topic fetcher drops synonyms that shadow a condition
    # already indexed, and drug_labels.csv is one of the corpora it checks against
    if args.source in ("drugs", "all"):
        run_drugs()
        
    if args.source in ("medlineplus", "all"):
        run_medlineplus()

if __name__ == "__main__":
    main()