# Tech Stack & Project Structure — Social Media Monitoring Platform

Architecture style: **Microservices**, one service per pipeline stage/domain, communicating via a message broker + shared REST/internal APIs.

---

## 1. Tech Stack

| Layer | Choice | Why (brief) |
|---|---|---|
| Frontend | **Next.js** (App Router, TypeScript) | User-specified; SSR for dashboards, good API-route support for BFF pattern |
| API Gateway | **FastAPI** (Python) | Async, auto OpenAPI docs, easiest integration with Python ML services downstream |
| Backend services | **FastAPI** (Python) per service | Same language as NLP/CV models (no serialization overhead across languages), async-friendly, industry standard for AI microservices |
| Relational DB | **Neon** (Postgres, serverless) | User-specified; branching + serverless scaling good fit for microservices dev |
| Vector DB | **Qdrant** (primary) / Chroma (dev fallback) | Qdrant: production-grade, filterable payloads, good for face-embedding + semantic search at scale; Chroma: simpler for local dev before Qdrant is provisioned |
| Search/aggregation | **Elasticsearch / OpenSearch** | Full-text + faceted filtering for dashboards, industry standard |
| Message broker | **Kafka** (or Redpanda for lighter self-hosted) | Decouples ingestion from processing; replayable; standard for event-driven microservices |
| Object storage | **MinIO** (S3-compatible) | Flagged video frames, media snapshots |
| Cache | **Redis** | Face-gallery lookups, rate-limit counters, session cache |
| Model serving | **Local models** (user-selected) via **Ollama / vLLM** (LLMs), **FastAPI + Torch/ONNX Runtime** (CV/NLP models) | Self-hosted inference wrapped in its own microservice per model family |
| Auth | **Clerk** or **Auth.js (NextAuth)** | Standard for Next.js apps, minimal setup |
| Container orchestration | **Docker Compose** (dev) → **Kubernetes** (prod) | Standard microservices deployment path |
| CI/CD | **GitHub Actions** | Industry default, per-service pipelines |
| Observability | **Prometheus + Grafana** (metrics), **Loki** (logs), **OpenTelemetry** (tracing) | Standard OSS observability stack for microservices |
| API contracts | **OpenAPI** (REST) + **Protobuf/gRPC** (internal service-to-service, optional) | REST for external/frontend-facing; gRPC optional for high-throughput internal calls (e.g., detection service) |

---

## 2. Project / Folder Structure

```
social-media-monitor/
│
├── frontend/                          # Next.js app
│   ├── app/
│   │   ├── (dashboard)/
│   │   │   ├── trends/page.tsx
│   │   │   ├── sentiment/page.tsx
│   │   │   ├── video-events/page.tsx
│   │   │   └── alerts/page.tsx
│   │   ├── api/                       # BFF routes (proxy to gateway)
│   │   └── layout.tsx
│   ├── components/
│   ├── lib/
│   ├── public/
│   ├── package.json
│   └── next.config.js
│
├── services/                          # All backend microservices
│   │
│   ├── api-gateway/                   # Single entry point for frontend
│   │   ├── app/
│   │   │   ├── routers/
│   │   │   ├── main.py
│   │   │   └── core/config.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── x-ingestion-service/           # X/Twitter API consumer
│   │   ├── app/
│   │   │   ├── x_client.py
│   │   │   ├── rules_manager.py
│   │   │   └── main.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── news-ingestion-service/        # RSS/HTML scrapers + YouTube Data API
│   │   ├── app/
│   │   │   ├── scrapers/
│   │   │   ├── youtube_client.py
│   │   │   └── main.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── text-nlp-service/              # Lang ID, NER, sentiment, topics, trends
│   │   ├── app/
│   │   │   ├── models/
│   │   │   │   ├── lang_id.py
│   │   │   │   ├── ner.py
│   │   │   │   ├── sentiment.py
│   │   │   │   └── topic_clustering.py
│   │   │   ├── orchestrator.py
│   │   │   └── main.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── video-ingestion-service/       # Stream connector, frame sampler
│   │   ├── app/
│   │   │   ├── stream_connector.py
│   │   │   ├── reconnect_manager.py
│   │   │   ├── frame_sampler.py
│   │   │   └── main.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── video-detection-service/       # Object/face/scene detection (GPU)
│   │   ├── app/
│   │   │   ├── object_detector.py
│   │   │   ├── face_detector.py
│   │   │   ├── face_recognizer.py
│   │   │   ├── scene_detector.py
│   │   │   ├── stt_client.py
│   │   │   ├── orchestrator.py
│   │   │   └── main.py
│   │   ├── Dockerfile.gpu
│   │   └── requirements.txt
│   │
│   ├── face-gallery-service/          # Enrollment + similarity search
│   │   ├── app/
│   │   │   ├── enrollment.py
│   │   │   ├── similarity_search.py
│   │   │   └── main.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── model-inference-service/       # Shared local LLM server (Ollama/vLLM wrapper)
│   │   ├── app/
│   │   │   ├── llm_client.py
│   │   │   └── main.py
│   │   ├── Dockerfile.gpu
│   │   └── requirements.txt
│   │
│   ├── alerting-service/              # Spike/threshold rules engine
│   │   ├── app/
│   │   └── Dockerfile
│   │
│   └── storage-service/               # Shared DB/ES/MinIO access layer (internal SDK, not exposed)
│       ├── app/
│       │   ├── postgres_client.py
│       │   ├── es_client.py
│       │   ├── vector_client.py       # Qdrant/Chroma
│       │   └── minio_client.py
│       └── requirements.txt
│
├── shared/                            # Shared Python package (installed into services)
│   ├── schemas/                       # Pydantic models shared across services
│   ├── events/                        # Kafka event contracts
│   └── utils/
│
├── infra/
│   ├── docker-compose.yml             # Local dev: all services + Postgres/Kafka/ES/MinIO/Redis
│   ├── docker-compose.gpu.yml         # GPU service overrides
│   ├── k8s/                           # Production manifests/Helm charts
│   └── terraform/                     # Cloud infra (if applicable)
│
├── docs/
│   ├── architecture.md
│   ├── plan.md
│   └── modules/
│
├── .github/
│   └── workflows/
│       ├── ci-frontend.yml
│       ├── ci-services.yml
│       └── deploy.yml
│
├── .env.example
└── README.md
```

---

## 3. Notes

- Each service owns its own `Dockerfile`, dependencies, and can be deployed/scaled independently (e.g., `video-detection-service` scales with GPU nodes; `x-ingestion-service` scales with CPU workers).
- `shared/` avoids duplicating Pydantic schemas and Kafka event contracts across services — single source of truth for inter-service message shapes.
- `storage-service` is not a public API — it's a shared internal client library other services import, keeping DB/vector-store access patterns consistent.
- GPU-bound services (`video-detection-service`, `model-inference-service`) are isolated with their own Dockerfiles/compose overrides so local dev without a GPU can still run everything else.