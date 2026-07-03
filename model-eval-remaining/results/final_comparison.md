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

## Part 2 — Media, Clustering & Summarization (4.3, 4.5–4.9)

_Appended after the 4.1/4.2/4.4 tables. Topic-modeling and summarization gold sets are PLACEHOLDER (rankings robust, absolute numbers provisional); STT is on a read-speech baseline; object/face detection are blocked on data._

### 4.3 Topic Modeling / Clustering

| Model | ARI | NMI | Fragmentation | Latency (gold fit / pool) |
|---|---|---|---|---|
| bertopic_mpnet | 0.354 | 0.627 | 6/6 stories split, 2 merged; purity 0.675 | 4.261s / 9.938s (3000 docs) |
| lda_gensim | 0.076 | 0.324 | 6/6 stories split, 6 merged; purity 0.450 | 0.025s / 7.543s (22646 docs) |

_Gold = PLACEHOLDER (40 hand-authored posts / 6 stories). Ranking is robust; absolute ARI/NMI are provisional until a real gold set is built._

**Recommendation — topic modeling:** default **`bertopic_mpnet`** (ARI 0.354). BERTopic (multilingual embeddings) decisively beats the bag-of-words LDA baseline on short, code-switched text; LDA is near-random (ARI ~0.08), so the CPU-cheap baseline is **not** a usable fast path here. BERTopic needs a GPU for embeddings, but topic modeling is a **batch** job off the live-stream path, so it does not bind the architecture.md §9.3 concurrency ceiling — schedule it so it doesn't overlap the real-time detectors.

### 4.5 Summarization

| Model | Type | ROUGE-L | BERTScore-F1 | Latency (ms/cluster) | Factuality flags |
|---|---|---|---|---|---|
| qwen3_4b_instruct | vllm | 0.345 | 0.923 | 645.700 | 5 |
| textrank_sumy | textrank | 0.132 | 0.854 | 14.100 | 0 |

_References = PLACEHOLDER drafts. BERTScore uses a multilingual model. Factuality flags on the Qwen run were all false positives on inspection (transliteration + common nouns) — no real hallucinations._

**Recommendation — summarization:** default **`qwen3_4b_instruct`** for quality (ROUGE-L 0.345, BERTScore 0.923). The extractive TextRank baseline (CPU, ~15 ms) is ~2.5× worse on ROUGE-L and returns raw code-switched posts, so it's only a degraded offline fallback. Qwen3-4B needs vLLM/GPU but is a **batch** job off the live path (§9.3 ceiling doesn't bind); run it after clustering.

### 4.6 Speech-to-Text

| Model | WER | CER | WER by lang (ur/en/mixed) | RTF (GPU) | VRAM (MiB) |
|---|---|---|---|---|---|
| fw_large_v3 | 0.219 | 0.080 | 0.219 / — / — | 0.132 | 4082 |
| fw_medium | 0.301 | 0.111 | 0.301 / — / — | 0.089 | 2226 |
| fw_small | 0.346 | 0.129 | 0.346 / — / — | 0.049 | 978 |

_WER/CER from the **forced `--language ur`** runs (best config); RTF/VRAM from the GPU float16 runs. Metrics are on the **FLEURS ur_pk read-speech** baseline — clean, monolingual Urdu, NOT the broadcast target._

**BLOCKED — needs:** real Pakistani **broadcast** clips (spontaneous, code-switched, noisy) with hand transcripts in `gold_sets/stt_gold.csv` for a domain-valid WER. FLEURS read speech is a floor; broadcast will be harder.

**Recommendation — STT:** **force / route the language** (auto-detect flips Urdu→Hindi, which wrecked WER >100%). Resource-efficient default **`fw_small`/`fw_medium`** (forced ur, tiny VRAM, RTF ≪1); escalate to **`fw_large_v3`** (best WER 0.219) for accuracy-critical streams. STT is on the **live-stream path**, so it competes for the single 24 GB GPU with the object/face detectors and vLLM — cross-check architecture.md §9.3: `fw_large_v3` (~4 GB, GPU RTF ~0.13) leaves headroom for only a few concurrent streams once detectors are co-resident; `fw_small`/`fw_medium` (~1–2 GB) allow more. For genuinely `mixed` clips a single forced language may hurt English segments — still open.

### 4.7 Object Detection

**BLOCKED — needs:** labelled Pakistani broadcast frames (`gold_sets/objects_gold.csv`: frame_path + boxes) and the YOLO/Detectron2 runs. COCO smoke-test is a generic check, not broadcast/logos.

_(Owned by a team member; not evaluated here. No fake result row.)_

**Recommendation — object detection:** pending data. Note this is on the **live-stream path** — architecture.md §9.3's GPU concurrency ceiling will gate the choice: heavier variants (e.g. YOLOv8x / InsightFace `buffalo_l`) cut the max concurrent streams per GPU, so the winning model must be checked against the real-time budget, not just accuracy.

### 4.8 Face Detection & Recognition

**BLOCKED — needs:** a WIDER FACE detection subset AND an enrollment gallery + gold test frames (`gold_sets/faces_gold.csv`, `datasets/faces/gallery/<id>/`). Biometric — legal review per architecture.md §9.1 before any production use.

_(Owned by a team member; not evaluated here. No fake result row.)_

**Recommendation — face detection:** pending data. Note this is on the **live-stream path** — architecture.md §9.3's GPU concurrency ceiling will gate the choice: heavier variants (e.g. YOLOv8x / InsightFace `buffalo_l`) cut the max concurrent streams per GPU, so the winning model must be checked against the real-time budget, not just accuracy.

### 4.9 Scene Detection

_Single candidate (PySceneDetect) — a sanity check, not a comparison; no accuracy metric without hand-labelled cuts._

- Last run: `datasets/scenes/synthetic_smoke.mp4` → **4 cuts** / 5 scenes, RTF **0.002** (detector=content, threshold=27.0).
- Harness verified on a synthetic clip (hard cuts at 3/6/9/12 s → detected exactly). Needs a real 1–3 min broadcast clip in `datasets/scenes/` for the actual spot-check.

**Recommendation — scene detection:** keep **PySceneDetect** (content detector). It's CPU-only with a negligible RTF (~0.002), so it runs comfortably alongside the GPU detectors and does **not** touch the architecture.md §9.3 GPU ceiling. Tune `--threshold` once real footage exists.

### Live-stream GPU budget (architecture.md §9.3)

Only **STT, object detection, and face detection** run on the live path and share the single 24 GB RTX 3090. Measured footprints so far: `fw_large_v3` ~4 GB / GPU-RTF ~0.13, `fw_medium` ~2.2 GB, `fw_small` ~1 GB. Topic modeling, summarization and scene detection are batch/CPU and don't bind the ceiling. **Flag:** stacking `fw_large_v3` STT with heavy object + face detectors (and any vLLM) on one 3090 will blow the concurrency ceiling — prefer smaller STT, or dedicate/scale GPUs, before committing the live path. Re-check against §9.3's exact per-GPU stream limit once object/face numbers exist.

