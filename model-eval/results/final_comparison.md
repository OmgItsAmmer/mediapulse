# Model Evaluation — Comparison Report

_Social media monitoring (Pakistani digital media). Hard subset = Roman Urdu / sarcasm / code-switched rows — the whole point of this eval._

### 4.1 Named Entity Recognition (NER)

| Model | Type | F1 (exact) (overall) | Hard-subset | Partial (hard) | Avg latency (ms) | Notes |
|---|---|---|---|---|---|---|
| qwen3_4b_instruct | vllm | 0.740 | 0.839 | 0.860 | 1169.5 | `Qwen/Qwen3-4B-Instruct-2507`; Already cached locally (~7.6GB); fits easily on 24GB. Used for quick vLLM smoke tests. |
| spacy_en_core_web_sm | spacy | 0.236 | 0.396 | 0.484 | 2.9 | `en_core_web_sm`; English-only baseline — no Urdu/Roman Urdu support (expected to be weak). |

_Overall = micro-average across 2 dataset(s): `datasets/ner/ner_public.csv`, `gold_sets/ner_gold.csv`. Hard-subset = Roman-Urdu / code-switched rows._

**Recommendation — ner:** winner on the hard subset is **`qwen3_4b_instruct`** (hard-subset 0.839, 1169.5 ms/call).

✅ `qwen3_4b_instruct` beats the fast baseline `spacy_en_core_web_sm` by **+0.443** hard-subset for ~**403×** the latency — escalation is justified for the hard subset; use the fast path for easy rows and escalate the rest (see `escalation_thresholds`).

### 4.2 Sentiment Analysis

| Model | Type | Accuracy (overall) | Hard-subset | Partial (hard) | Avg latency (ms) | Notes |
|---|---|---|---|---|---|---|
| qwen3_4b_instruct | vllm | 0.622 | 0.882 | — | 37.7 | `Qwen/Qwen3-4B-Instruct-2507`; Already cached locally (~7.6GB). Used for the current sentiment run. |
| xlmr_sentiment_baseline | hf_pipeline | 0.530 | 0.294 | — | 3.0 | `cardiffnlp/twitter-xlm-roberta-base-sentiment`; Multilingual XLM-R sentiment head (resource-efficient fast path). Runs on conda-base torch. |

_Overall = micro-average across 2 dataset(s): `datasets/sentiment/combined.csv`, `gold_sets/sentiment_gold.csv`. Hard-subset = sarcasm=True rows._

**Recommendation — sentiment:** winner on the hard subset is **`qwen3_4b_instruct`** (hard-subset 0.882, 37.7 ms/call).

✅ `qwen3_4b_instruct` beats the fast baseline `xlmr_sentiment_baseline` by **+0.588** hard-subset for ~**13×** the latency — escalation is justified for the hard subset; use the fast path for easy rows and escalate the rest (see `escalation_thresholds`).

### 4.4 Language Identification

| Model | Type | Accuracy (overall) | Hard-subset | Partial (hard) | Avg latency (ms) | Notes |
|---|---|---|---|---|---|---|
| qwen3_4b_instruct | vllm | 0.539 | 0.319 | — | 48.1 | `Qwen/Qwen3-4B-Instruct-2507`; Already cached locally (~7.6GB). Used for the current langid run. |
| langdetect | langdetect | 0.486 | 0.000 | — | 1.3 | `langdetect` |
| fasttext_lid176 | fasttext | 0.462 | 0.000 | — | 0.0 | `datasets/langid/lid.176.bin`; Download lid.176.bin from official fastText releases before running. |

_Overall = micro-average across 2 dataset(s): `datasets/langid/combined.csv`, `gold_sets/langid_gold.csv`. Hard-subset = Roman-Urdu / code-switched rows._

**Recommendation — langid:** winner on the hard subset is **`qwen3_4b_instruct`** (hard-subset 0.319, 48.1 ms/call).

✅ `qwen3_4b_instruct` beats the fast baseline `langdetect` by **+0.319** hard-subset for ~**36×** the latency — escalation is justified for the hard subset; use the fast path for easy rows and escalate the rest (see `escalation_thresholds`).

