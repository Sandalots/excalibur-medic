# EXCALIBUR-Medic
An offline clinical reference device. A 1B model, distilled from a 4-bit 8B medical
teacher and quantised to 770 MB, answers questions from a BM25 index over 19,411 NIH,
FDA and MedlinePlus documents — on a Raspberry Pi 5 with 2 GB of RAM, with no network at
inference.

It looks things up. It does not know them: every answer is generated from a retrieved
passage, and where no suitable passage exists the system says so rather than guessing.

Built on a MacBook M3 Pro (18 GB) with Apple MLX.

## What it is for
Answering reference questions that name a condition or a drug — *what are the symptoms of
hypothyroidism*, *who should not take warfarin*, *is lisinopril safe in pregnancy*. It is
a lookup tool, not a triage assistant, and it declines questions about a specific person's
situation.

## Three artefacts
| File | What it does |
|---|---|
| [`excalibur.ipynb`](excalibur.ipynb) | Builds the model: chain-of-thought generation from the teacher, quality gating, LoRA fine-tuning, fuse → GGUF → imatrix Q4_K_M |
| [`excalibur_inference.py`](excalibur_inference.py) | The deployed runtime: corpus, BM25 index, five guards, verbatim path, and the interface to `llama-server`. Standard library only |
| [`benchmark_pi.sh`](benchmark_pi.sh) | Measures the device: throughput, peak memory, index cost, end-to-end latency; `--energy` adds power and a thermal soak |

Supporting: [`deploy_to_pi.sh`](deploy_to_pi.sh) copies the payload to the board,
[`macintosh/mac_infer.sh`](macintosh/mac_infer.sh) runs the same runtime on the Mac, and
[`scripts/`](scripts/) fetches and verifies the corpus.

## Models and corpus
| | |
|---|---|
| Teacher | `mlx-community/Llama3-OpenBioLLM-8B`, 4-bit |
| Student | `mlx-community/Llama-3.2-1B-Instruct-bf16` |
| Deployed | `excalibur-medic-1b-Q4_K_M.gguf`, 770 MB |
| Corpus | MedQuAD (NIH), openFDA drug labels, MedlinePlus health topics — 19,411 indexed documents |

The combined corpus and the index are derived on the device at first run and cached, so
neither is shipped.

## Getting started
Full setup for both machines, including the `llama.cpp` builds, is in
[`INSTALLATION.md`](INSTALLATION.md).

```bash
# on the Pi, after deploying
cd ~/excalibur

python3 excalibur_inference.py                      # interactive
python3 excalibur_inference.py --lite               # prompt only
python3 excalibur_inference.py -q "What is the dose of metformin?"
python3 excalibur_inference.py --stats              # per-answer telemetry
```

In session, `?` lists the commands, `stats` toggles the timing block and `exit` quits.

```bash
./benchmark_pi.sh            # threads, memory, index, end-to-end
./benchmark_pi.sh --energy   # ~50 min: idle floor, joules per query, thermal soak
```

## Layout
```
excalibur.ipynb              build the model
excalibur_inference.py       the deployed runtime
benchmark_pi.sh              measure the device
deploy_to_pi.sh              copy to the board
reset.sh                     delete generated artefacts
macintosh/mac_infer.sh       run the runtime on the Mac
scripts/                     corpus fetching and verification
datasets/, artifacts/        source data and build outputs
INSTALLATION.md              full setup
```