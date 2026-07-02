# Model Evaluation — Comparison Report

_Social media monitoring (Pakistani digital media). Hard subset = Roman Urdu / sarcasm / code-switched rows — the whole point of this eval._

### 4.1 Named Entity Recognition (NER)

| Model | Type | F1 (exact) (overall) | Hard-subset | Partial (hard) | Avg latency (ms) | Notes |
|---|---|---|---|---|---|---|
| qwen3_4b_instruct | vllm | 0.740 | 0.839 | 0.860 | 1169.5 | `Qwen/Qwen3-4B-Instruct-2507`; Already cached locally (~7.6GB); fits easily on 24GB. Used for quick vLLM smoke tests. |
| spacy_en_core_web_sm | spacy | 0.236 | 0.396 | 0.484 | 2.9 | `en_core_web_sm`; English-only baseline — no Urdu/Roman Urdu support (expected to be weak). |

_Overall = micro-average across 2 dataset(s): `datasets/ner/ner_public.csv`, `gold_sets/ner_gold.csv`. Hard-subset = Roman-Urdu / code-switched rows only._

**Recommendation — ner:** winner on the hard subset is **`qwen3_4b_instruct`** (hard-subset 0.839, 1169.5 ms/call).

✅ `qwen3_4b_instruct` beats the fast baseline `spacy_en_core_web_sm` by **+0.443** hard-subset for ~**403×** the latency — escalation is justified for the hard subset; use the fast path for easy rows and escalate the rest (see `escalation_thresholds`).

### Not yet evaluated

No results found for: **sentiment, langid** — run the corresponding eval scripts (Prompts 2 / 3), then re-run this.

