# model-eval — NLP Model Evaluation Harness

Evaluation harness for a **social-media monitoring platform for Pakistani digital
media** (X/Twitter, news sites, YouTube). Before building the NLP pipeline, this
project shortlists and benchmarks candidate models for three tasks by running them
locally on **vLLM** and comparing accuracy **and** latency.

| Task                | What it does                                               |
| ------------------- | ---------------------------------------------------------- |
| **4.1 NER**         | Named-entity recognition — people, orgs, brands, locations |
| **4.2 Sentiment**   | positive / negative / neutral, must handle **sarcasm**     |
| **4.4 Language ID** | english / urdu / roman_urdu / mixed (code-switched)        |

**The core challenge — and the whole point of this eval:** standard NLP tools fail on
**Roman Urdu** (Urdu written in Latin script) and **code-switched** Urdu/English text.
Language-ID tools tag Roman Urdu as English; sentiment models miss sarcasm. So every
eval reports accuracy **separately on the hard subset** (Roman Urdu / sarcasm /
code-switched), not just an aggregate that hides the failure.

---

## TL;DR — Headline results

Runs so far use the locally-cached **Qwen3-4B-Instruct-2507** as the LLM (a _floor_ for
what the LLM path delivers; the configured **Qwen3-8B** should match or beat it).
Gold sets are **placeholder** data (small) — treat numbers as directional until the
gold sets are expanded and hand-verified.

| Task            | Metric (hard subset)                      | Baseline                     | Qwen3-4B | Baseline latency | LLM latency |
| --------------- | ----------------------------------------- | ---------------------------- | -------- | ---------------- | ----------- |
| **NER**         | exact F1 on Roman-Urdu/code-switched gold | spaCy **0.40**               | **0.84** | ~3 ms            | ~1170 ms    |
| **Sentiment**   | accuracy on **sarcasm** rows              | XLM-R **0.29**               | **0.88** | ~3 ms            | ~38 ms      |
| **Language ID** | accuracy on roman_urdu (gold)             | fastText/langdetect **0.00** | **0.70** | ~1 ms            | ~48 ms      |

**Consistent finding:** baselines collapse on the hard subset; the LLM recovers most of
it. The latency cost is steep for NER (~400×, long JSON output) but cheap for sentiment
and langid (~13–36×, single-word output). This is what the `escalation_thresholds` in
`config.yaml` are for: run the cheap model on easy rows, escalate the hard ones.

The canonical, auto-generated comparison is in
[`results/final_comparison.md`](results/final_comparison.md).

---

## Repository structure

```
model-eval/
├── README.md                  # this file
├── config.yaml                # models per task, vLLM URL, escalation thresholds
├── requirements.txt           # eval-venv deps (see notes about torch/vLLM)
├── .gitignore
├── scripts/
│   ├── serve_and_eval.sh      # serve ONE vLLM model → wait for /health → eval → teardown
│   ├── prepare_ner_data.py    # normalize UNER + MK-PUCIT   → datasets/ner/ner_public.csv
│   ├── prepare_sentiment_data.py #                          → datasets/sentiment/combined.csv
│   ├── prepare_langid_data.py #                             → datasets/langid/combined.csv
│   ├── make_ner_gold.py       # generate gold_sets/ner_gold.csv        (offsets auto-computed)
│   ├── make_sentiment_gold.py # generate gold_sets/sentiment_gold.csv  (sarcasm-focused)
│   ├── make_langid_gold.py    # generate gold_sets/langid_gold.csv     (40 rows, 10/class)
│   ├── eval_ner.py            # NER eval:      spaCy + vLLM, exact/partial/untyped F1
│   ├── eval_sentiment.py      # sentiment:     XLM-R + vLLM, acc + macro-F1 + sarcasm-only acc
│   ├── eval_langid.py         # language ID:   fastText + langdetect + vLLM, per-class + confusion matrix
│   ├── run_ner_eval.sh        # loop all NER models × datasets
│   ├── run_sentiment_eval.sh  # loop all sentiment models (uses conda-base python for torch)
│   ├── run_langid_eval.sh     # loop all langid models
│   └── compare_results.py     # aggregate results/*.summary.json → results/final_comparison.md
├── gold_sets/                 # hand-authored PLACEHOLDER test sets  ← COMMIT THESE
│   ├── ner_gold.csv           #   30 rows, Roman-Urdu/code-switched, entity spans
│   ├── sentiment_gold.csv     #   30 rows, sarcasm-focused (+ sarcasm bool)
│   └── langid_gold.csv        #   40 rows, 10 per class incl. deceptive Roman-Urdu
├── datasets/                  # downloaded + derived data  ← GITIGNORED (regenerable)
│   ├── ner/       raw/ (UNER, MK-PUCIT) + ner_public.csv
│   ├── sentiment/ raw/ (roman.csv, urdu.tsv) + combined.csv
│   └── langid/    lid.176.bin (131 MB) + combined.csv
├── results/                   # eval outputs
│   ├── *.summary.json         #   aggregate metrics per (model × dataset)  ← COMMIT
│   ├── *.csv                  #   per-row predictions (run artifacts)      ← gitignored
│   └── final_comparison.md    #   the report                              ← COMMIT
├── logs/                      # vLLM serve logs, install logs  ← gitignored
└── venv/                      # eval virtualenv                ← gitignored
```

---

## Environment

- **OS/HW:** Ubuntu, single **NVIDIA RTX 3090 (24 GB)**.
- **Two Python environments (intentional):**
  1. **`venv/`** (Python 3.13) — the eval harness deps: `openai`, `pandas`,
     `scikit-learn`, `seqeval`, `spacy` + `en_core_web_sm`, `requests`, `tqdm`,
     `pyyaml`, `langdetect`, `fasttext-predict`, `ucimlrepo`. Used for NER + langid
     evals and all data prep.
  2. **conda base** (Python 3.13, ships with vLLM 0.23) — has `torch 2.11` +
     `transformers 5.6` + `openai`. Used to (a) **serve** vLLM models and (b) run
     the **sentiment XLM-R baseline** (needs torch). Path:
     `/home/temp/miniconda3/bin/python`.
- vLLM serves **one** LLM at a time on the single GPU. The eval scripts talk to it
  over HTTP via the OpenAI-compatible client.

### Recreate the environment

```bash
python3 -m venv venv
venv/bin/python -m pip install -r requirements.txt
venv/bin/python -m spacy download en_core_web_sm
# torch/transformers for the sentiment baseline live in the conda-base env (with vLLM).
```

---

## How to run

### 0. Serve helper — `scripts/serve_and_eval.sh`

Serves one model, waits for `/health` (fails fast if the process dies), runs an eval
script against it, then **gracefully tears the server down** (kills the whole process
tree even on Ctrl-C). vLLM logs go to `logs/vllm_<model>_<ts>.log`.

```bash
./scripts/serve_and_eval.sh Qwen/Qwen3-8B eval_ner.py
# env overrides: PORT, MAX_MODEL_LEN, GPU_MEM_UTIL, HEALTH_TIMEOUT, VENV_PY, EVAL_LIMIT
```

### 1. Prepare data + gold sets (regenerable)

```bash
venv/bin/python scripts/prepare_ner_data.py         # downloads UNER + MK-PUCIT
venv/bin/python scripts/prepare_sentiment_data.py   # reuses roman.csv + urdu.tsv
venv/bin/python scripts/prepare_langid_data.py
venv/bin/python scripts/make_ner_gold.py
venv/bin/python scripts/make_sentiment_gold.py
venv/bin/python scripts/make_langid_gold.py
```

### 2. Run evals (quick smoke = set EVAL_LIMIT)

```bash
CONDA=/home/temp/miniconda3/bin/python

# NER (spaCy baseline needs no GPU; Qwen served via helper)
EVAL_LIMIT=15 venv/bin/python scripts/eval_ner.py --model spacy_en_core_web_sm
EVAL_LIMIT=15 ./scripts/serve_and_eval.sh Qwen/Qwen3-4B-Instruct-2507 eval_ner.py

# Sentiment (baseline needs torch → conda-base python)
EVAL_LIMIT=200 $CONDA scripts/eval_sentiment.py --model xlmr_sentiment_baseline
EVAL_LIMIT=200 VENV_PY=$CONDA ./scripts/serve_and_eval.sh Qwen/Qwen3-4B-Instruct-2507 eval_sentiment.py

# Language ID (all in venv)
venv/bin/python scripts/eval_langid.py --model fasttext_lid176
venv/bin/python scripts/eval_langid.py --model langdetect
./scripts/serve_and_eval.sh Qwen/Qwen3-4B-Instruct-2507 eval_langid.py

# Or loop everything per task:
./scripts/run_ner_eval.sh
./scripts/run_sentiment_eval.sh
./scripts/run_langid_eval.sh
```

### 3. Build the comparison report

```bash
venv/bin/python scripts/compare_results.py            # → results/final_comparison.md
```

---

### 4.1 NER — `eval_ner.py`

- **Models:** spaCy `en_core_web_sm` (English-only baseline), Qwen via vLLM.
- **Public data:** **UNER** (1,637 sentences, was UTF-16LE, inline `<TYPE>` tags) +
  **MK-PUCIT** (9.27M-token stream with _no sentence boundaries_ → segmented into
  entity-safe ~25-token windows, sampled) — both from the `mirfan899/Urdu` repo.
  Normalized to `[text, entities(JSON), lang, source]`, types mapped to
  PERSON/ORG/LOCATION/BRAND/DATE/MISC.
- **Metrics:** entity-level micro **exact** (span+type), **partial** (same type +
  char overlap), **untyped** (span only — diagnoses type confusion). Reported overall
  and on the Roman-Urdu/code-switched hard subset.
- **Key design:** LLM-provided char offsets are _ignored and re-derived_ by searching
  the source text (LLMs miscount characters); malformed JSON is caught & counted.

### 4.2 Sentiment — `eval_sentiment.py`

- **Models:** `cardiffnlp/twitter-xlm-roberta-base-sentiment` (multilingual baseline,
  runs on conda-base torch/GPU), Qwen via vLLM.
- **Public data:** Roman Urdu `roman.csv` (20,110) + Urdu `urdu.tsv` (1,000), shuffled,
  normalized to `[text, label, lang, source]`.
- **Prompt:** few-shot, explicitly sarcasm-aware ("label the INTENT, not the literal
  words"); output = one label word.
- **Metrics:** accuracy + macro-F1 overall, **and a separate accuracy/macro-F1 on
  `sarcasm=True` rows** (never averaged in), plus per-class F1 and per-language accuracy.

### 4.4 Language ID — `eval_langid.py`

- **Models:** fastText `lid.176.bin` (via `fasttext-predict`), `langdetect`, Qwen via vLLM.
- **Public data:** roman_urdu + urdu (2,000). **ERUPD** (arXiv 2412.17562) is
  **Kaggle-gated** → the `english` and `mixed` classes live in the gold set, which is
  the balanced 4-class benchmark.
- **Metrics:** overall accuracy + macro-F1, **per-class accuracy**, and a full
  **confusion matrix** (with an `other` bucket for ISO codes like `hi`/`id` that
  baselines emit). Baselines map `en→english`, `ur→urdu`, else→`other` — so they
  _structurally cannot_ output roman_urdu/mixed, which the matrix exposes directly.

---

## Results & interpretation

- **NER:** on the hard gold subset spaCy manages exact F1 0.40 (finds Latin-script
  spans but mislabels types) and ~0.03 on Urdu script (can't read it). Qwen3-4B: 0.84
  (hard) / 0.64 (Urdu). Escalation clearly justified, but ~400× slower (long JSON).
- **Sentiment:** the baseline scores **0.29 on sarcasm — worse than random** (it reads
  literal words). Qwen3-4B: **0.88**. On _non_-sarcastic Roman-Urdu polarity the gap
  nearly closes (0.53 vs 0.58) — the LLM's edge is specifically nuance/sarcasm. LLM is
  only ~13× slower here.
- **Language ID:** both baselines score a **flat 0.00** on roman_urdu and mixed
  (fastText sends 706/1000 Roman-Urdu rows to English). Qwen3-4B reaches 0.70
  (roman_urdu) / 0.55 (hard subset) on the clean gold set. `mixed` is genuinely hard
  even for the LLM (0.40) — the roman_urdu↔mixed boundary is fuzzy.

**Caveats:** (1) gold sets are **placeholder** (30–40 rows) — expand & hand-verify
before quoting; (2) results use **4B**;
(3) sarcasm-subset macro-F1 looks low despite high accuracy because those rows are
label-imbalanced; (4) the langid `combined` set is short/noisy Roman-Urdu fragments —
trust the balanced gold numbers.

---

## Data sources & attribution

- **UNER, MK-PUCIT, roman.csv, urdu.tsv:** [mirfan899/Urdu](https://github.com/mirfan899/Urdu)
- **fastText lid.176:** fastText official releases
- **ERUPD** (English↔Roman Urdu): [arXiv 2412.17562](https://arxiv.org/abs/2412.17562) (Kaggle-gated; not bundled)
- **Precisely Xtreme** Roman-Urdu sentiment: [arXiv 2003.05443](https://arxiv.org/abs/2003.05443) (no public data file found)
- **Sentiment baseline:** `cardiffnlp/twitter-xlm-roberta-base-sentiment`
- **LLM:** Qwen (`Qwen/Qwen3-4B-Instruct-2507`)

---
