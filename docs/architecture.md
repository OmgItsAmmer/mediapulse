# Social Media Monitoring Platform — Architecture (Phase 1: Twitter/X + Digital Media Focus)

> Scope note: This document covers the **current focus** — Social/Digital Media Monitoring on **Twitter/X**, Pakistani digital media sources (news sites, YouTube news channels), **and live-stream video ingestion with object/face detection**, which is now a **high-priority current-phase item** (not deferred). Full video metadata enrichment depth (STT, OCR, diarization, scene/topic summarization on video) remains later-phase, but the live-stream + object/face detection slice is being built now alongside the text pipeline.

---

## 1. High-Level Architecture Diagram

```
                                ┌───────────────────────────────────────────┐
                                │            INGESTION LAYER                 │
                                │                                             │
   ┌───────────────┐           │  ┌──────────────┐   ┌────────────────────┐ │
   │  X (Twitter)   │──────────┼─▶│ X API v2      │   │  Digital Media      │ │
   │  API v2        │           │  │ (filtered     │   │  Scrapers/Fetchers  │ │
   └───────────────┘           │  │ stream +      │   │  (RSS, HTML, RTSP/  │ │
                                │  │ search)       │   │  HLS for YT live)   │ │
   ┌───────────────┐           │  └──────┬───────┘   └──────────┬─────────┘ │
   │ Pakistani News │──────────┼─────────┘                      │           │
   │ Sites (RSS/    │           │                                │           │
   │ HTML)          │           │        ┌───────────────────────┘           │
   └───────────────┘           │        ▼                                    │
   ┌───────────────┐           │  ┌──────────────────┐                       │
   │ YouTube News   │──────────┼─▶│  Message Queue     │                     │
   │ Channels (API +│           │  │  (Kafka/Redpanda)  │                     │
   │ live chat)     │           │  └─────────┬─────────┘                     │
   └───────────────┘           └────────────┼──────────────────────────────┘
                                              ▼
                                ┌───────────────────────────────────────────┐
                                │          PROCESSING LAYER (Workers)        │
                                │                                             │
                                │  ┌────────────┐  ┌────────────────────┐    │
                                │  │ Text        │  │ Entity Extraction   │    │
                                │  │ Normalizer  │─▶│ (NER: people, orgs, │    │
                                │  │ (lang detect,│  │ brands, locations)  │    │
                                │  │ dedupe)     │  └──────────┬──────────┘    │
                                │  └────────────┘             │               │
                                │                              ▼               │
                                │  ┌────────────┐  ┌────────────────────┐    │
                                │  │ Sentiment   │◀─┤ Local LLM           │    │
                                │  │ Classifier  │  │ (summarization,     │    │
                                │  └─────┬──────┘  │ topic labeling)      │    │
                                │        │          └──────────┬──────────┘    │
                                │        ▼                     ▼               │
                                │  ┌──────────────────────────────────────┐   │
                                │  │  Topic Clustering (BERTopic)          │   │
                                │  │  Trend Detection (burst/volume calc)  │   │
                                │  └──────────────────┬───────────────────┘   │
                                └─────────────────────┼───────────────────────┘
                                                        ▼
                                ┌───────────────────────────────────────────┐
                                │             STORAGE LAYER                  │
                                │  ┌────────────┐  ┌─────────────────────┐  │
                                │  │ PostgreSQL  │  │ Vector DB (pgvector/ │  │
                                │  │ (structured │  │ Qdrant) for semantic │  │
                                │  │ metadata)   │  │ search               │  │
                                │  └────────────┘  └─────────────────────┘  │
                                │  ┌──────────────────────────────────────┐ │
                                │  │ Elasticsearch/OpenSearch (full-text,  │ │
                                │  │ filters, aggregations, dashboards)    │ │
                                │  └──────────────────────────────────────┘ │
                                └─────────────────────┬───────────────────────┘
                                                        ▼
                                ┌───────────────────────────────────────────┐
                                │              SERVING LAYER                 │
                                │  ┌────────────┐  ┌─────────────────────┐  │
                                │  │ REST/GraphQL│  │ Dashboard (React +   │  │
                                │  │ API         │  │ charts: trends,      │  │
                                │  │             │  │ sentiment, mentions) │  │
                                │  └────────────┘  └─────────────────────┘  │
                                │  ┌──────────────────────────────────────┐ │
                                │  │ Alerting (keyword spike, sentiment    │ │
                                │  │ drop, new mention thresholds)         │ │
                                │  └──────────────────────────────────────┘ │
                                └───────────────────────────────────────────┘

   ══════════════ FUTURE PHASE (not current focus) ══════════════
   ┌───────────────────────────────────────────────────────────┐
   │  VIDEO/AUDIO ENRICHMENT PIPELINE                            │
   │  Upload/Stream → STT → Diarization → OCR → Face/Object/Logo │
   │  Detection → Scene Detection → Summarization → Metadata      │
   │  Index (feeds into same Storage/Serving layers above)        │
   └───────────────────────────────────────────────────────────┘
```

---

## 2. Pipeline Stages (Current Focus: Twitter/X + Digital Media Text)

1. **Ingestion**
   - X API v2 filtered stream (keyword/hashtag rules) + periodic search for backfill/trending.
   - RSS/HTML fetchers for Pakistani news sites (see Section 6).
   - YouTube Data API for news channel metadata, comments, and live chat (where applicable).
2. **Queueing** — buffer raw events for async processing, decouple ingestion rate from processing rate.
3. **Normalization** — language detection (Urdu/English/Roman Urdu mix is common in Pakistani social content), deduplication, spam/bot filtering.
4. **Entity Extraction (NER)** — people, organizations, brands, locations mentioned.
5. **Sentiment Analysis** — positive/negative/neutral, ideally with confidence score.
6. **Topic Modeling / Clustering** — group related posts into topics/stories.
7. **Trend Detection** — volume/velocity spikes per keyword, hashtag, or entity.
8. **Live-Stream Video Ingestion** — pull frames from RTSP/RTMP/HLS/YouTube live sources (via FFmpeg) at a sampling interval.
9. **Object / Face Detection on Frames** — detect and (optionally) identify people, objects, logos in sampled frames in near-real-time.
10. **Storage** — structured metadata (Postgres), full-text + aggregation (Elasticsearch), semantic search (vector DB), detected-entity/frame metadata linked back to source stream + timestamp.
11. **Serving** — dashboard, API, alerting (text + visual detections combined).

---

## 3. Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Ingestion (X) | X API v2 (filtered stream + search) | Official, compliant, real-time capable via streaming endpoint |
| Ingestion (news) | `feedparser` (RSS), `Scrapy`/`Playwright` (HTML), YouTube Data API | Covers structured (RSS) and unstructured (HTML-only sites) sources |
| Queue | Kafka or Redpanda | Decouples ingestion spikes from processing; replayable for reprocessing with improved models later |
| Workers | Python (FastAPI workers) + Celery/RQ or Kafka consumers | Standard, well-supported async processing pattern |
| NLP (NER) | spaCy (transformer pipeline) | Fast, production-grade, good multilingual support extensions |
| NLP (Sentiment/Topics/Summarization) | Local LLM (Qwen3-8B/14B, quantized) via Ollama/vLLM | Nuanced sentiment on mixed Urdu/English/Roman Urdu content beats rule-based classifiers |
| Topic clustering | BERTopic | Well-suited to noisy short-form social text |
| Structured storage | PostgreSQL | Reliable, relational, easy joins across entities/mentions/sources |
| Semantic search | pgvector (start) → Qdrant (scale) | pgvector avoids extra infra at MVP; Qdrant when query volume grows |
| Full-text/aggregation | Elasticsearch or OpenSearch | Best-in-class for filters, faceted search, dashboard aggregations |
| Dashboard | React + Recharts/Chart.js | Standard, flexible for trend/sentiment visualizations |
| Alerting | Custom rules engine on top of stream processor (or ksqlDB on Kafka) | Real-time spike/sentiment-drop detection without polling |

---

## 4. Video / Live-Stream Scanning Architecture (Current Phase, High Priority)

This is a **separate real-time pipeline** running alongside the text pipeline (Section 2), specifically for scanning live video streams (RTSP/RTMP/HLS/YouTube live) for objects, faces, and scene events. It shares the same Storage/Serving layers but has its own ingestion and processing path because live video has fundamentally different latency and compute constraints than text.

### 4.1 Video Pipeline Diagram

```
   ┌────────────────────────────────────────────────────────────────┐
   │                     LIVE VIDEO SOURCES                          │
   │  RTSP (CCTV/NVR) │ RTMP │ HLS/M3U8 │ HTTP(S) │ YouTube Live      │
   └──────────────────────────────┬───────────────────────────────────┘
                                    ▼
                        ┌─────────────────────┐
                        │  Stream Connector     │
                        │  (FFmpeg / GStreamer) │
                        │  - reconnect/retry    │
                        │  - resolution normalize│
                        └──────────┬────────────┘
                                    ▼
                        ┌─────────────────────┐
                        │  Frame Sampler        │
                        │  (N fps → 1 frame per │
                        │  X seconds, tunable)  │
                        └──────────┬────────────┘
                                    ▼
              ┌─────────────────────┴─────────────────────┐
              ▼                                             ▼
   ┌─────────────────────┐                       ┌─────────────────────┐
   │  Video Track          │                       │  Audio Track         │
   │  (sampled frames)      │                       │  (extracted stream)  │
   └──────────┬────────────┘                       └──────────┬────────────┘
              ▼                                                 ▼
   ┌─────────────────────────────┐                  ┌─────────────────────┐
   │  Object Detection (YOLOv8n)  │                  │  Speech-to-Text       │
   │  → objects, logos             │                  │  (Faster-Whisper)     │
   └──────────┬────────────────────┘                  └──────────┬────────────┘
              ▼                                                   ▼
   ┌─────────────────────────────┐                  ┌─────────────────────┐
   │  Face Detection (RetinaFace)  │                  │  (Optional) Diarization│
   └──────────┬────────────────────┘                  │  (pyannote)           │
              ▼                                        └──────────┬────────────┘
   ┌─────────────────────────────┐                                │
   │  Face Match Against Gallery   │                                │
   │  (InsightFace embeddings,     │                                │
   │  cosine similarity)           │                                │
   └──────────┬────────────────────┘                                │
              ▼                                                     │
   ┌─────────────────────────────┐                                │
   │  Scene Change Detection       │                                │
   │  (PySceneDetect)              │                                │
   └──────────┬────────────────────┘                                │
              │                                                     │
              └───────────────────┬─────────────────────────────────┘
                                    ▼
                        ┌─────────────────────┐
                        │  Event Aggregator     │
                        │  (timestamp-align all │
                        │  detections + confid- │
                        │  ence scores)          │
                        └──────────┬────────────┘
                                    ▼
                        ┌─────────────────────┐
                        │  Escalation Router     │
                        │  low-confidence /      │
                        │  watchlist match →     │
                        │  Quality-Efficient      │
                        │  models (full InsightFace,│
                        │  YOLOv8x, Whisper large)│
                        └──────────┬────────────┘
                                    ▼
                     ═══════════ into Storage Layer ═══════════
                     PostgreSQL (event metadata + timestamps)
                     Elasticsearch (searchable frame/event index)
                     Object storage (S3/MinIO) for flagged frame snapshots
```

### 4.2 Stage-by-Stage Breakdown

1. **Stream Connector** — FFmpeg (or GStreamer for lower-level control) connects to the source, handles protocol differences (RTSP/RTMP/HLS/HTTP), and — critically — manages reconnect/retry logic since real-world streams drop frequently.
2. **Frame Sampler** — pulls one frame every N seconds (tunable; e.g., every 1-3 seconds) rather than processing every frame, since running detection on every frame of a live 25-30fps stream is not feasible on a single GPU.
3. **Object Detection** — runs YOLOv8n on sampled frames to detect objects/logos in near-real-time.
4. **Face Detection + Matching** — RetinaFace detects faces in-frame; matched faces get embedded (ArcFace) and compared against the enrollment gallery (see earlier face-recognition discussion) via cosine similarity.
5. **Audio Track (parallel)** — audio is extracted from the same stream and run through Faster-Whisper for live captioning/transcription, optionally with diarization if multiple speakers matter for the use case.
6. **Scene Detection** — flags shot/scene changes so events can be timestamped against meaningful video segments, not just raw frame numbers.
7. **Event Aggregator** — merges detections from all tracks (object, face, audio, scene) into a single timestamped event record per stream.
8. **Escalation Router** — this is where the resource/quality-efficient split (Section 5) actually gets applied at runtime: anything low-confidence or matching a watchlist gets re-run through the heavier, more accurate model instead of trusting the fast first-pass result.
9. **Storage** — structured events go to PostgreSQL, searchable index to Elasticsearch, and any flagged frame images get saved to object storage (S3-compatible, e.g. MinIO if self-hosted) for human review/audit.

### 4.3 Tech Stack Specific to Video Scanning

| Component | Technology | Why |
|---|---|---|
| Stream ingestion | FFmpeg | Handles RTSP/RTMP/HLS/HTTP/UDP natively, industry standard, well-documented |
| Frame extraction | OpenCV (`cv2.VideoCapture`) or FFmpeg pipe | Simple, GPU-accelerable, integrates directly with Python detection models |
| Object detection | YOLOv8 (Ultralytics) | Fast inference, easy to fine-tune for custom logo/object classes, runs well on a single GPU |
| Face detection | RetinaFace | Purpose-built for detection speed, pairs naturally with ArcFace for recognition |
| Face recognition | InsightFace (ArcFace embeddings) | Enrollment/gallery-matching approach (no per-person fine-tuning needed), industry-standard accuracy |
| Face/vector similarity search | FAISS or pgvector | Fast nearest-neighbor lookup against the enrolled face gallery, scales to hundreds of tracked individuals |
| Speech-to-text | Faster-Whisper | 4x speed over vanilla Whisper, same accuracy, fits comfortably on a 24GB GPU alongside vision models |
| Scene detection | PySceneDetect | CPU-only, doesn't compete for GPU with the vision models |
| Frame/event storage | MinIO (S3-compatible) + PostgreSQL | Cheap self-hosted object storage for flagged frames; Postgres for structured event metadata |
| Orchestration | Python asyncio workers, one process per stream (or worker pool with stream-affinity) | Keeps each live stream's pipeline isolated so one stream's failure/reconnect doesn't block others |

### 4.4 Why This Is Architecturally Separate from the Text Pipeline

- **Latency requirements differ** — text processing can tolerate queue delays of seconds/minutes; live video scanning needs near-real-time frame processing or the pipeline falls behind the live stream.
- **GPU residency matters** — vision models (YOLO, RetinaFace, InsightFace) benefit from staying loaded in GPU memory continuously per active stream, unlike text models that can be invoked more statelessly per batch.
- **Failure modes are different** — a dropped RTSP connection needs reconnect logic; a failed API call in the text pipeline just retries from the queue. These require different resilience patterns.
- **Concurrency ceiling is hardware-bound, not just throughput-bound** — the number of live streams you can scan simultaneously is capped by GPU VRAM/compute (see Section 9.3), whereas text pipeline throughput scales more linearly with worker count.



Every task below lists the model, whether it's the **Resource-Efficient** (cheap/fast, low VRAM, high volume) or **Quality-Efficient** (best accuracy, higher compute cost) choice, and why. Default routing pattern: run Resource-Efficient on all incoming data; escalate flagged/ambiguous/high-priority items (tracked VIPs, viral spikes, low-confidence detections) to the Quality-Efficient model.

### 4.1 Named Entity Recognition (People, Orgs, Brands, Locations)

| Model | Efficiency Type | Justification |
|---|---|---|
| spaCy `en_core_web_sm` (CPU) | 🟢 Resource-Efficient | Extremely fast, no GPU needed, handles the bulk of clear-cut English entity mentions at high volume |
| spaCy `en_core_web_trf` (+ custom Urdu/Roman-Urdu fine-tune) | 🟡 Balanced | Production-proven, extensible to custom entity types (Pakistani political parties, local brands) via `spacy-transformers` fine-tuning |
| Local LLM (Qwen3-8B), prompted for NER | 🔴 Quality-Efficient | Handles code-switched Urdu/English/Roman Urdu far better zero-shot than spaCy; use only for low-confidence/ambiguous cases due to per-call latency |
| XLM-RoBERTa fine-tuned on Urdu NER datasets | 🔴 Quality-Efficient | Best pure-accuracy option for Urdu-heavy content, but requires labeled data + fine-tuning investment |

### 4.2 Sentiment Analysis

| Model | Efficiency Type | Justification |
|---|---|---|
| Fine-tuned mBERT / XLM-RoBERTa sentiment head | 🟢 Resource-Efficient | Cheap per-inference, good enough for high-confidence/clear-sentiment posts — first-pass filter for the bulk of volume |
| Local LLM (Qwen3-8B/14B, few-shot prompted) | 🔴 Quality-Efficient | Handles sarcasm, code-switching, political nuance common in Pakistani content — routes to this only for ambiguous/low-confidence cases from the fast-path model |
| Cloud LLM (Claude/GPT-class API) | 🔴 Quality-Efficient (highest ceiling) | Best raw accuracy available, but per-call cost doesn't scale to monitoring volume — reserve for QA/spot-checking, not the main pipeline |

### 4.3 Topic Modeling / Clustering

| Model | Efficiency Type | Justification |
|---|---|---|
| LDA (classic) | 🟢 Resource-Efficient | Cheap compute, but weaker on short-text (tweets) and doesn't handle code-switching well — acceptable only as a low-cost fallback |
| BERTopic + multilingual embeddings (`paraphrase-multilingual-mpnet-base-v2`) | 🔴 Quality-Efficient | Handles noisy, short-form, mixed-language social text well — default recommended choice given Pakistani content patterns |

### 4.4 Language Detection

| Model | Efficiency Type | Justification |
|---|---|---|
| `fasttext` lid.176 | 🟢 Resource-Efficient | Extremely fast, tiny, CPU-only, good baseline accuracy — default for all incoming text |
| `langdetect` / `polyglot` | 🟢 Resource-Efficient | Simpler, slightly less accurate — fine for low-volume prototyping only |
| Custom fine-tuned classifier for Roman Urdu | 🔴 Quality-Efficient | Needed because Roman Urdu (Urdu in Latin script) is routinely misclassified as English by all standard language-ID tools — a real, specific pain point for Pakistani content that justifies the extra investment |

### 4.5 Summarization

| Model | Efficiency Type | Justification |
|---|---|---|
| Extractive (TextRank) | 🟢 Resource-Efficient | No GPU, near-instant, good for quick digest-style summaries at high volume |
| Local LLM (Qwen3-8B or Llama 3.1-8B) | 🟡 Balanced | Good abstractive quality, lower VRAM, fits alongside other concurrent GPU workloads |
| Local LLM (Qwen3-14B quantized) | 🔴 Quality-Efficient | Best quality-to-cost ratio for summarizing topic clusters into human-readable briefs when VRAM headroom allows |

### 4.6 Speech-to-Text (Live Stream Audio) — *Current Phase*

| Model | Efficiency Type | Justification |
|---|---|---|
| Faster-Whisper (small/medium) | 🟢 Resource-Efficient | Low VRAM (~2GB), good enough accuracy for real-time captioning of live streams |
| Faster-Whisper (large-v3) | 🔴 Quality-Efficient | ~5-6GB VRAM, best accuracy — use for archived/high-priority streams where latency is less critical |
| NVIDIA Parakeet/Canary | 🔴 Quality-Efficient | Top open ASR leaderboard accuracy; better fit once on multi-GPU/cloud infra than single-GPU live constraints |

### 4.7 Object Detection (Live Stream Frames) — *Current Phase, High Priority*

| Model | Efficiency Type | Justification |
|---|---|---|
| YOLOv8n / YOLOv8s (nano/small) | 🟢 Resource-Efficient | Very low latency, runs comfortably on a single GPU alongside other live-stream tasks — right fit for real-time frame sampling |
| YOLOv11 / YOLOv8x (large) | 🔴 Quality-Efficient | Higher accuracy for logo/object detail, but slower inference — better for archived footage than real-time streams |
| Detectron2 | 🔴 Quality-Efficient | More accurate on complex scene understanding, but heavier — not ideal for real-time; suited to batch/archive analysis |

### 4.8 Face Detection & Recognition (Live Stream Frames) — *Current Phase, High Priority*

| Model | Efficiency Type | Justification |
|---|---|---|
| RetinaFace (mobile/lightweight variant) | 🟢 Resource-Efficient | Fast face detection suitable for real-time frame sampling on live streams |
| InsightFace (`buffalo_l`, ArcFace + RetinaFace) | 🔴 Quality-Efficient | 99.86% LFW accuracy, industry-standard embedding approach — use full-resolution/enrollment matching for flagged frames or archived footage rather than every live frame |
| DeepFace (multi-backend wrapper) | 🟡 Balanced | Lets you A/B test backbones without rewriting pipeline code; good for evaluation before committing to InsightFace in production |

### 4.9 Scene Detection (Live Stream / Video Segmentation) — *Current Phase*

| Model | Efficiency Type | Justification |
|---|---|---|
| PySceneDetect | 🟢 Resource-Efficient | CPU-only, lightweight, sufficient for detecting shot/scene changes to timestamp events in a live feed |

---

### Practical Routing Strategy (applies across all tables above)

Run the **🟢 Resource-Efficient** model as the default on all incoming data/frames (handles the large majority of volume cheaply, and in real time for live streams). Escalate only flagged, ambiguous, or high-priority content — tracked VIP faces, viral spikes, low-confidence detections, or specific watchlisted objects/logos — to the **🔴 Quality-Efficient** model. This keeps the single-GPU live-stream pipeline (object + face detection, both high priority) running in real time while still getting high-accuracy results where it matters most.

---

## 7. Pakistani Digital Media Sources (Monitoring Targets)

### 6.1 News Websites (RSS/HTML scrape-friendly)
- **Dawn.com** — Pakistan's largest English-language outlet, has RSS feeds
- **The News International** (thenews.com.pk)
- **Tribune Express** (tribune.com.pk)
- **Geo.tv** / Geo News (geo.tv)
- **Samaa.tv**
- **ARY News** (arynews.tv)
- **Dunya News** (dunyanews.tv)
- **Bol News** (bolnews.com)
- **92 News** (92newshd.tv)
- **GNN** (gnn.tv)
- **Daily Jang** (jang.com.pk) — largest Urdu-language daily
- **Nawa-i-Waqt** (nawaiwaqt.com.pk) — Urdu daily
- **Express News Urdu** (express.pk)

### 6.2 YouTube News Channels (live + VOD, via YouTube Data API)
- ARY News official channel
- SAMAA TV — runs 24/7 live news streaming
- BOL News — large subscriber base, live news + talk shows
- Hum News — 24/7 live coverage
- Dawn News
- Dunya News
- GNN
- Aggregator channels compiling talk shows from multiple outlets (e.g., "Top Pakistani News"-style channels compiling shows like Nadeem Malik Live, Black & White)

### 6.3 Social Platforms
- **X (Twitter)** — primary source for real-time political/public discourse, hashtag trends
- **YouTube comments + live chat** — useful for sentiment on breaking news/live coverage
- **Facebook Pages** (via Meta's Graph API where accessible) — still dominant in Pakistan for news distribution, though API access is more restricted than X

### 6.4 Practical Notes on Scraping/Access
- Most Pakistani news sites **do not have strict anti-scraping measures** but check `robots.txt` and terms of use per-site before building scrapers — build a compliance checklist per source.
- **YouTube live streams** are technically accessible via `yt-dlp`/HLS extraction for audio/video pipeline (future phase), but Data API is the correct path for metadata, comments, and live chat at MVP stage — avoid scraping YouTube directly where the official API covers your need.
- **Facebook** remains a major distribution channel for Pakistani news outlets but has the most restrictive API access of the major platforms — deprioritize until X and news-site coverage are solid.
- Roman Urdu content is heavy across all these sources' comment sections — budget NLP effort specifically for this (see Section 4.4).

---

---

## 8. Open Questions / Next Decisions

- [ ] Finalize keyword/hashtag/entity watchlist for MVP (people, orgs, brands to track)
- [ ] Decide Elasticsearch vs. OpenSearch (licensing considerations)
- [ ] Confirm X API budget tier given monitoring volume (see cost analysis — filtered stream vs. search-heavy usage)
- [ ] Legal review: data retention policy for scraped news content and social posts, and for face recognition/biometric processing
- [ ] Roman Urdu NLP: build or source a labeled dataset for fine-tuning language ID + sentiment
- [ ] Confirm GPU capacity plan (single 3090 vs. cloud burst) for concurrent live-stream count target

---

## 9. Potential Hurdles — Where We Can Get Stuck, and Why

### 8.1 Data Access & Legal
- **X API cost scaling** — pay-per-use pricing ($0.005/read, $0.010 for user/profile data) means monitoring dozens of keywords continuously can blow past budget fast; the 2M read cap forces a jump to Enterprise pricing (~$42K/mo) if exceeded. **Risk: cost surprises mid-scale, not at MVP.**
- **Facebook/Meta access** — most restrictive of the major platforms; if a future requirement needs FB coverage, expect slow approval processes and limited data compared to X.
- **News site scraping compliance** — some Pakistani outlets may change site structure, add anti-bot measures, or restrict via `robots.txt` without notice, breaking scrapers silently. **Risk: silent data gaps** that aren't obvious until someone notices missing coverage.
- **Face recognition legal exposure** — identifying specific named individuals is biometric processing; depending on jurisdiction and use case (public figures vs. private individuals), this could require consent/retention policies not yet defined. **This needs a legal review before production, not after.**
- **YouTube live stream extraction (yt-dlp/HLS)** — technically works today, but is fragile against platform-side changes (YouTube regularly tweaks stream delivery to block scrapers); official Data API doesn't cover raw video/audio access, only metadata.

### 8.2 Language & NLP
- **Roman Urdu is the single biggest NLP risk.** No standard model handles it well out of the box — language ID misclassifies it as English, sentiment models trained on English/Urdu miss code-switched sarcasm, and NER on Roman Urdu names/entities has weak coverage. This isn't a solved problem; expect to build/label custom data.
- **Code-switching mid-sentence** (Urdu + English + Roman Urdu in one post) breaks most off-the-shelf pipelines that assume one language per document.
- **Sarcasm and political nuance** — Pakistani political discourse is heavy on sarcasm/irony, which even LLM-based sentiment struggles with; expect meaningfully lower accuracy here than in benchmark numbers.
- **No labeled training data yet** — every "quality-efficient" model that needs fine-tuning (Roman Urdu NER, sentiment) requires a labeled dataset that doesn't currently exist and will take real time/budget to build.

### 8.3 Live-Stream Video Pipeline (Current Phase, High Priority)
- **Single-GPU concurrency ceiling** — one RTX 3090 can realistically handle only a handful of concurrent live streams running object + face detection simultaneously before frame-sampling rate has to drop or queuing delays appear. **This directly limits how many streams can be monitored live at once without additional GPU capacity.**
- **Stream instability** — RTSP/RTMP/HLS sources from real-world cameras/broadcasts drop frames, reconnect, or shift resolution/bitrate unpredictably; the ingestion layer needs robust reconnect/retry logic or it will silently stop processing a feed.
- **Face recognition accuracy in real-world conditions** — benchmark accuracy (e.g., InsightFace's 99.86% on LFW) is measured on clean, front-facing photos; live broadcast footage has poor lighting, motion blur, side angles, and compression artifacts, all of which degrade real accuracy meaningfully below benchmark numbers.
- **Latency vs. accuracy tradeoff in real time** — the resource-efficient models (YOLOv8n, RetinaFace-lite) are chosen specifically because quality-efficient models (YOLOv8x, full InsightFace) are too slow for real-time frame-by-frame processing on a single GPU; if real-time accuracy requirements turn out to be stricter than expected, the current single-GPU setup won't be enough.
- **Frame sampling rate tradeoffs** — sampling too infrequently risks missing short-duration events (a logo/face appearing briefly); sampling too frequently overloads the GPU — this threshold needs empirical tuning per use case, not a fixed default.

### 8.4 Infrastructure & Scale
- **Kafka/queue operational overhead** — running Kafka (or Redpanda) reliably requires real operational know-how (partitioning, consumer lag monitoring, retention config); underestimating this is a common early-stage stall point.
- **Vector DB migration path** — starting on pgvector is fine for MVP, but migrating to Qdrant later (once query volume grows) means re-indexing and potential downtime if not planned for from the start.
- **Elasticsearch/OpenSearch resource cost** — full-text + aggregation at scale is memory-hungry; underestimating cluster sizing is a common source of production slowdowns once data volume grows past prototype scale.
- **No cloud burst plan yet** — the current single-GPU setup is great for MVP/dev, but there's no defined path yet for scaling to cloud GPUs (RunPod, Lambda, etc.) when concurrent stream/volume needs exceed local hardware — this should be decided before it becomes an urgent blocker.

### 8.5 Team & Process
- **Model sprawl risk** — with 9+ tasks each having 2-4 model options (resource vs. quality-efficient), without a clear owner/decision log, the team can end up with inconsistent choices across environments (dev vs. staging vs. prod) or duplicated effort re-evaluating the same tradeoffs.
- **Evaluation criteria undefined** — "accuracy" thresholds for triggering escalation from resource-efficient to quality-efficient models (e.g., sentiment confidence score, face-match similarity threshold) haven't been benchmarked yet against real Pakistani social/media data — these need empirical tuning, not assumed defaults, before the routing logic in Section 4 can work reliably in production.