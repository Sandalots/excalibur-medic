# The EXCALIBUR Medic Model
An offline clinical reference device. A 1B model, distilled from a 4-bit 8B medical
teacher and quantised to 770 MB, answers questions from a BM25 index over 19,411 NIH,
FDA and MedlinePlus documents, targeted to run on a Raspberry Pi 5 with 2 GB of RAM, with no network at
inference.

It looks things up. Every answer is generated from a retrieved
passage, and where no suitable passage exists the system says so rather than guessing.

Built on a MacBook M3 Pro (18 GB) with Apple MLX, also tested to work on a M4 Mac mini (16 GB).

Answering reference questions that name a condition or a drug, for example the following potential user queries; *what are the symptoms of
hypothyroidism*, *who should not take warfarin*, *is lisinopril safe in pregnancy*.

## Three implementing artefacts
| Filename | Purpose |
|---|---|
| [`excalibur.ipynb`](excalibur.ipynb) | Builds the excalibur model: chain-of-thought generation from the teacher, quality gating, LoRA fine-tuning, fuse to GGUF to imatrix Q4_K_M |
| [`excalibur_inference.py`](excalibur_inference.py) | The deployed runtime inference script for the Pi 5: corpus, BM25 index, five guards, verbatim path, and the interface to `llama-server`. Standard library only |
| [`benchmark_pi.sh`](benchmark_pi.sh) | Measures the Pi 5 device: throughput, peak memory, index cost, end-to-end latency; `--energy` adds power and a thermal sustained soak to check for thermal throttling|

Supporting: [`deploy_to_pi.sh`](deploy_to_pi.sh) copies the model payload to the pi 5 board.

## Models and corpus used
| | |
|---|---|
| Teacher | `mlx-community/Llama3-OpenBioLLM-8B`, 4-bit |
| Student | `mlx-community/Llama-3.2-1B-Instruct-bf16` |
| Deployed | `excalibur-medic-1b-Q4_K_M.gguf`, 770 MB |
| Corpus | MedQuAD (NIH), openFDA drug labels, MedlinePlus health topics — 19,411 indexed medical documents |

The combined corpus and the index are derived on the device at first run and cached, so
neither is shipped. First run index setup costs roughly a second or two.

## Getting started
Full setup for both machines, including the `llama.cpp` builds, is in
[`INSTALLATION.md`](INSTALLATION.md).

```bash
# on the Pi, after deploying
cd ~/excalibur

python3 excalibur_inference.py                      # interactive
python3 excalibur_inference.py --lite               # prompt only
python3 excalibur_inference.py -q "What is the dose of metformin?" # supply a single medical query directly at command line
python3 excalibur_inference.py --stats              # per-answer telemetry
```

In session, `?` lists the commands, `stats` toggles the timing block and `exit` quits.

```bash
./benchmark_pi.sh            # threads, memory, index, end-to-end
./benchmark_pi.sh --energy   # ~50 min: idle floor, joules per query, thermal soak
```
