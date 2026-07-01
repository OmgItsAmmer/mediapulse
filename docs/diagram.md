flowchart TB

    subgraph SOURCES["📡 SOURCES"]
        direction LR
        X["X / Twitter API v2<br/>(filtered stream + search)"]
        NEWS["Pakistani News Sites<br/>(Dawn, Geo, ARY, Samaa, Jang...)<br/>RSS/HTML"]
        YT["YouTube News Channels<br/>(Data API: videos, comments, live chat)"]
    end

    subgraph INGEST["⚙️ INGESTION"]
        direction LR
        FETCH["Scrapers / API Fetchers"]
        Q["Message Queue<br/>(Kafka / Redpanda)"]
    end

    subgraph PROCESS["🧠 PROCESSING (Workers)"]
        direction TB
        NORM["Text Normalizer<br/>(lang detect, dedupe, spam filter)"]
        LANGID{"Roman Urdu /<br/>Urdu / English?"}
        NER["NER<br/>(spaCy → LLM fallback)<br/>people, orgs, brands, locations"]
        SENT["Sentiment Classifier<br/>(mBERT fast-path → LLM for ambiguous)"]
        TOPIC["Topic Clustering<br/>(BERTopic + multilingual embeddings)"]
        TREND["Trend Detection<br/>(volume/velocity spikes)"]
        LLM["Local LLM<br/>(Qwen3 8B/14B — summarization,<br/>topic labeling, ambiguous cases)"]

        NORM --> LANGID
        LANGID --> NER
        NER --> SENT
        SENT --> TOPIC
        TOPIC --> TREND
        SENT -.ambiguous.-> LLM
        NER -.low confidence.-> LLM
        TOPIC -.summarize.-> LLM
    end

    subgraph STORE["🗄️ STORAGE"]
        direction LR
        PG[("PostgreSQL<br/>structured metadata")]
        VEC[("pgvector / Qdrant<br/>semantic search")]
        ES[("Elasticsearch /<br/>OpenSearch<br/>full-text + aggregations")]
    end

    subgraph SERVE["📊 SERVING"]
        direction LR
        API["REST / GraphQL API"]
        DASH["Dashboard<br/>(React + charts)<br/>trends · sentiment · mentions"]
        ALERT["Alerting Engine<br/>keyword spikes, sentiment drops"]
    end

    SOURCES --> FETCH --> Q --> PROCESS
    TREND --> PG
    TREND --> VEC
    TREND --> ES
    LLM --> PG

    PG --> API
    VEC --> API
    ES --> API
    API --> DASH
    API --> ALERT

    classDef future fill:#f5f5f5,stroke:#999,stroke-dasharray: 5 5,color:#666;
    subgraph FUTURE["🎥 FUTURE PHASE — Video/Audio Enrichment (not current focus)"]
        direction LR
        VUP["Upload / Live Stream<br/>(RTSP/RTMP/HLS/YouTube)"]
        STT["STT + Diarization<br/>(Whisper, pyannote)"]
        VIS["OCR + Face/Object/Logo<br/>Detection (PaddleOCR, InsightFace, YOLO)"]
        SCENE["Scene Detection +<br/>Summarization"]
        VUP --> STT --> VIS --> SCENE --> PG
    end
    class FUTURE,VUP,STT,VIS,SCENE future