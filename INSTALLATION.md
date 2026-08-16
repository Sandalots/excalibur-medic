# Installation
Two machines. The **Mac** trains the model and builds the artefact; the **Pi** runs it.
Nothing is trained on the Pi 5. Inference is solely done on-device on the Pi 5, as-well as the models benchmarking measuring.

| | Mac | Raspberry Pi 5 |
|---|---|---|
| does | CoT generation, SFT, fuse to GGUF to quantise | serves answers, offline |
| needs | Apple silicon, 18 GB RAM | Pi 5 2 GB, headless, ~2 GB free |
| takes | ~13 h, resumable | ~25 min, mostly compiling llama-cpp |

Part A produces the deployable excalibur model artefact; Part B puts it on the device and measures it.

## Part A · The Mac
### A1 · llama.cpp
Build it first — the notebook calls its converter and quantiser.

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build ~/llama.cpp/build -j 10
```

Elsewhere is fine — set `LLAMA_CPP=/path/to/llama.cpp` and both the notebook and
`excalibur_inference.py` honour it.

### A2 · Environment
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e ~/llama.cpp/gguf-py
.venv/bin/python -m ipykernel install --user --name excalibur \
    --display-name "EXCALIBUR (.venv)"

.venv/bin/python -c "import mlx_lm, torch, gguf; print('ok')"
```

### A3 · Run the notebook
```bash
jupyter lab excalibur.ipynb        # kernel: EXCALIBUR (.venv)
```

Run All.

| stage | how long |
|---|---|
| dependencies, config, split by `focus_area` | seconds |
| load the 4-bit teacher, apply three patches | ~2 min, 4.9 GB download |
| **CoT generation, 6,000 samples** | **~11.65 h** |
| quality gate — expect ~91% kept | seconds |
| behaviour data | ~1 min |
| **SFT, 2,500 LoRA steps** | **~46 min** |
| held-out eval, abstention | ~5 min |
| fuse to GGUF to imatrix to quantise to verify | ~4 min |
| corpus, analysis, figures, run summary | ~2 min |

**CoT generation is resumable.** It appends to `artifacts/synthetic/cot_full.jsonl` and skips what is
already there.

**Do not run the behaviour-data section alone.** It reads `sft_train.jsonl`, which the
quality gate writes; running it twice without the gate between appends the behaviour
examples again. Run All is always safe.

### A4 · Rebuild without retraining
Rerun the two export sections; each step skips when its output is newer than its input.

```bash
./reset.sh          # remove generated artefacts, keep the fetched corpus
```

### A5 · Rebuild the corpus (optional)
The fetched corpus ships with the repository.

```bash
.venv/bin/python scripts/fetcher.py all     # openFDA + NLM -> CSVs + aliases
```

Re-fetching gives you today's openFDA and MedlinePlus, so figures will land near the
recorded ones rather than on them, so will drift slightly.

### A6 · Verify
No Pi 5 or GPU acceleration needed.

```bash
.venv/bin/python scripts/verify_retrieval.py    # guards, verbatim path, retrieval
.venv/bin/python scripts/evaluate.py ceiling    # oracle retrieval
.venv/bin/python scripts/evaluate.py ablate     # base vs SFT, identical retrieval
```

Figures are rebuilt by the notebook's figures section.

## Part B · The Raspberry Pi 5
### B1 · Build llama.cpp
```bash
sudo apt update && sudo apt install -y cmake g++ git python3
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build ~/llama.cpp/build -j4 --target llama-server llama-cli
```

`-DGGML_NATIVE=ON` enables the Cortex-A76's NEON and FP16 instructions — worth 30–40% of
throughput. `-j4` rather than `-j$(nproc)`: four compilers is near the memory ceiling on
a 2 GB board. Expect ~20 minutes.

### B2 · Deploy
```bash
ssh pi@raspberrypi.local 'mkdir -p ~/excalibur/artifacts'

./deploy_to_pi.sh                              # asks for the address
HOST=pi@raspberrypi.local ./deploy_to_pi.sh    # or supply it, skipping the prompt
```

It asks for the Pi's address as `user@host`, then once for the SSH password, and reuses
that one connection for every transfer. The model is skipped when its checksum already
matches, so a re-deploy after a corpus change moves kilobytes.

```
~/excalibur/
├── excalibur_inference.py
├── benchmark_pi.sh
├── datasets/medquad.csv
└── artifacts/
    ├── excalibur-medic-1b-Q4_K_M.gguf
    ├── drug_labels.csv
    ├── drug_aliases.json
    ├── medlineplus.csv
    └── topic_aliases.json
```

The combined corpus and BM25 index are derived on the device at first run and cached,
and rebuilt whenever a source file changes.

### B3 · Run
```bash
cd ~/excalibur
python3 excalibur_inference.py                    # interactive
python3 excalibur_inference.py --lite             # prompt only
python3 excalibur_inference.py -q "What is the dose of metformin?"
python3 excalibur_inference.py --stats            # per-answer telemetry
```

In session: `?` for commands, `stats` to toggle timings, `exit` to quit.

### B4 · Measure
```bash
./benchmark_pi.sh            # threads, peak memory, BM25 on ARM, end-to-end
./benchmark_pi.sh --energy   # ~50 min: idle floor, joules per query, thermal soak
```

`--energy` requires a Pi 5; it reads the PMIC via `vcgencmd pmic_read_adc` and stops
within seconds if that returns nothing.

The last two benchmark runs are also reproduced within the table from the supplied measurement subdirectory that contains the latest two runs plus an archive of older runs.
