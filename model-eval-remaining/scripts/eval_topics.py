#!/usr/bin/env python3
"""
eval_topics.py — evaluate ONE topic-modeling / clustering model (Prompt 5, Task 3).

Models (config.yaml, models.topics):
  - gensim_lda : bag-of-words LDA baseline (CPU). Install: venv/bin/pip install gensim
  - bertopic   : BERTopic + a multilingual sentence-embedding model
                 (default paraphrase-multilingual-mpnet-base-v2). Needs torch +
                 sentence-transformers (conda base) + `pip install bertopic`.

Topic modeling is UNSUPERVISED, so we don't score exact-label matches. Instead:

  ACCURACY  — measured on the small LABELLED gold set (gold_sets/topics_gold.csv,
              `story_id` column, ~40 code-switched posts / 6 stories). We fit the
              model on those posts with num_topics = story count and compute:
                * Adjusted Rand Index (ARI)      vs story_id
                * Normalized Mutual Information (NMI) vs story_id
                * FRAGMENTATION — how many gold stories got split across >1 predicted
                  topic, and how many predicted topics MERGED >1 gold story. For a
                  handful of news stories this matters more than the raw metric.

  LATENCY   — measured separately on the FULL pool (datasets/topics/pool.csv): the
              wall-clock fit + transform time at real corpus scale. This is the
              throughput number; it is NOT scored for accuracy.

Both models are forced to num_topics clusters (LDA num_topics; BERTopic via a
KMeans(k) cluster backend) so the head-to-head is apples-to-apples at the same k.
BERTopic's HDBSCAN auto-discovery is available with --bertopic-cluster hdbscan
(useful on the pool; it collapses to mostly outliers on 40 gold rows, so it is not
the default for the gold accuracy run).

Outputs:
  results/topics_<model>_<ts>.csv          gold row -> predicted topic id
  results/topics_<model>_<ts>.summary.json  ARI / NMI / fragmentation / latency

Run:
  venv/bin/python scripts/eval_topics.py --model lda_gensim
  conda run -n base python scripts/eval_topics.py --model bertopic_mpnet
  # quick pool-latency timing on a subsample:
  ... --model bertopic_mpnet --pool-limit 2000
"""
import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42

# Light stopword lists for the LDA bag-of-words path (English + common Roman-Urdu
# function words). Not exhaustive — LDA is the baseline, kept deliberately simple.
EN_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that",
    "these", "those", "at", "by", "from", "as", "so", "than", "too", "very", "can",
    "will", "just", "not", "no", "up", "out", "off", "over", "all", "any", "has",
    "have", "had", "do", "did", "does", "you", "your", "we", "our", "they", "their",
    "he", "she", "his", "her", "i", "me", "my", "us", "them",
}
ROMAN_STOP = {
    "hai", "hain", "ho", "ho gaya", "gaya", "gayi", "gaye", "ka", "ki", "ke", "ko",
    "se", "mein", "main", "aur", "ye", "yeh", "wo", "woh", "kya", "na", "nahi", "nahin",
    "par", "pe", "bhi", "to", "tha", "thi", "the", "jo", "ne", "hi", "yaar", "sab",
    "kar", "raha", "rahe", "rahi", "diya", "di", "liya", "aaj", "phir",
}
STOPWORDS = EN_STOP | ROMAN_STOP

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text):
    toks = _TOKEN.findall(text.lower())
    return [t for t in toks if len(t) >= 2 and t not in STOPWORDS]


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


# ============================ predictors ===================================
class LDAPredictor:
    """gensim LDA. fit_predict builds a fresh model each call and returns
    (topic-id per doc, elapsed_seconds) covering the full fit + transform."""

    type = "gensim_lda"

    def __init__(self, spec):
        try:
            import gensim  # noqa: F401
        except ImportError:
            sys.exit("ERROR: gensim not installed. Run: venv/bin/pip install gensim")
        self.spec = spec

    def fit_predict(self, texts, k):
        from gensim import corpora
        from gensim.models import LdaModel
        t0 = time.perf_counter()
        tokenized = [tokenize(t) for t in texts]
        dictionary = corpora.Dictionary(tokenized)
        # keep the vocab usable even on a tiny gold set
        if len(dictionary) > 20:
            dictionary.filter_extremes(no_below=1, no_above=0.6)
        corpus = [dictionary.doc2bow(toks) for toks in tokenized]
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k,
                       random_state=SEED, passes=10, iterations=100,
                       alpha="auto", eval_every=None)
        preds = []
        for bow in corpus:
            dist = lda.get_document_topics(bow, minimum_probability=0.0)
            preds.append(max(dist, key=lambda x: x[1])[0] if dist else -1)
        return preds, (time.perf_counter() - t0)


class BERTopicPredictor:
    """BERTopic + a multilingual sentence-transformer. The embedding model is loaded
    once and reused; each fit_predict builds a fresh BERTopic on top of it."""

    type = "bertopic"

    def __init__(self, spec, cluster_backend):
        try:
            import bertopic  # noqa: F401
        except ImportError:
            sys.exit("ERROR: bertopic not installed. From conda base (has torch + "
                     "sentence-transformers): pip install bertopic")
        from sentence_transformers import SentenceTransformer
        self.spec = spec
        self.cluster_backend = cluster_backend        # "kmeans" | "hdbscan"
        self.embed_model_id = spec["model_id"]
        print(f"  loading embeddings: {self.embed_model_id} ...")
        self.embedder = SentenceTransformer(self.embed_model_id)

    def _umap(self, n):
        from umap import UMAP
        return UMAP(n_neighbors=min(15, max(2, n - 1)), n_components=5,
                    min_dist=0.0, metric="cosine", random_state=SEED)

    def fit_predict(self, texts, k):
        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer
        t0 = time.perf_counter()
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        kwargs = dict(embedding_model=self.embedder,
                      vectorizer_model=CountVectorizer(stop_words="english"),
                      umap_model=self._umap(len(texts)),
                      calculate_probabilities=False, verbose=False)
        if self.cluster_backend == "kmeans":
            from sklearn.cluster import KMeans
            kwargs["hdbscan_model"] = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        # else: BERTopic default HDBSCAN (auto topic count; may emit -1 outliers)
        topic_model = BERTopic(**kwargs)
        topics, _ = topic_model.fit_transform(texts, embeddings)
        return list(topics), (time.perf_counter() - t0)


# ============================ metrics ======================================
def clustering_metrics(story_ids, preds):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    return (round(float(adjusted_rand_score(story_ids, preds)), 4),
            round(float(normalized_mutual_info_score(story_ids, preds)), 4))


def fragmentation(story_ids, preds):
    """A gold story is FRAGMENTED if its posts land in >1 predicted topic.
    A predicted topic is MERGED if it contains posts from >1 gold story."""
    by_story, by_topic = defaultdict(list), defaultdict(list)
    for s, p in zip(story_ids, preds):
        by_story[s].append(p)
        by_topic[p].append(s)

    per_story, n_frag = {}, 0
    for s, ps in sorted(by_story.items()):
        topics = sorted(set(ps))
        per_story[str(s)] = {"n_posts": len(ps), "predicted_topics": topics,
                             "n_topics": len(topics)}
        if len(topics) > 1:
            n_frag += 1

    per_topic, n_merged = {}, 0
    for p, ss in sorted(by_topic.items()):
        stories = sorted(set(ss))
        per_topic[str(p)] = {"n_posts": len(ss), "gold_stories": stories,
                             "n_stories": len(stories)}
        if len(stories) > 1:
            n_merged += 1

    purity = round(sum(max(Counter(ss).values()) for ss in by_topic.values()) / len(preds), 4)
    return {
        "n_gold_stories": len(by_story),
        "n_predicted_topics": len(by_topic),
        "n_fragmented_stories": n_frag,
        "n_merged_topics": n_merged,
        "cluster_purity": purity,
        "per_story": per_story,
        "per_topic": per_topic,
    }


# ============================ io ===========================================
def load_config(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_model(cfg, model_arg):
    for spec in cfg.get("models", {}).get("topics", []):
        if model_arg in (spec.get("name"), spec.get("model_id")):
            return dict(spec)
    return None


def load_gold(path):
    texts, story_ids, labels = [], [], []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            t = (r.get("text") or "").strip()
            sid = (r.get("story_id") or "").strip()
            if not t or not sid:
                continue
            texts.append(t)
            story_ids.append(int(sid) if sid.isdigit() else sid)
            labels.append(r.get("story_label", ""))
    return texts, story_ids, labels


def load_pool(path, limit):
    texts = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            t = (r.get("text") or "").strip()
            if t:
                texts.append(t)
            if limit and len(texts) >= limit:
                break
    return texts


# ============================ main =========================================
def main():
    ap = argparse.ArgumentParser(description="Evaluate one topic-modeling model.")
    ap.add_argument("--model", default=os.environ.get("EVAL_MODEL"),
                    help="config name: lda_gensim | bertopic_mpnet")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--gold", default=os.path.join(HERE, "gold_sets/topics_gold.csv"))
    ap.add_argument("--pool", default=os.path.join(HERE, "datasets/topics/pool.csv"))
    ap.add_argument("--num-topics", type=int, default=0,
                    help="k for LDA / KMeans. 0 = number of distinct gold story_id.")
    ap.add_argument("--pool-limit", type=int, default=int(os.environ.get("POOL_LIMIT", "0")),
                    help="cap pool rows for the latency run (0 = full pool).")
    ap.add_argument("--bertopic-cluster", choices=["kmeans", "hdbscan"], default="kmeans",
                    help="BERTopic cluster backend (kmeans forces k; hdbscan auto-discovers).")
    ap.add_argument("--skip-pool", action="store_true",
                    help="skip the pool latency run (accuracy on gold only).")
    args = ap.parse_args()

    if not args.model:
        sys.exit("ERROR: no model. Use --model lda_gensim|bertopic_mpnet or set $EVAL_MODEL.")
    cfg = load_config(args.config)
    spec = resolve_model(cfg, args.model)
    if not spec:
        avail = [s["name"] for s in cfg.get("models", {}).get("topics", [])]
        sys.exit(f"ERROR: model '{args.model}' not in config.topics. Available: {avail}")

    if not os.path.exists(args.gold):
        sys.exit(f"ERROR: gold set missing: {args.gold}. Run scripts/make_topics_gold.py first.")
    texts, story_ids, labels = load_gold(args.gold)
    if not texts:
        sys.exit(f"ERROR: no usable rows in {args.gold}.")

    k = args.num_topics or len(set(story_ids))
    results_dir = os.path.join(HERE, cfg.get("paths", {}).get("results", "results"))
    os.makedirs(results_dir, exist_ok=True)

    print(f"Model: {spec['name']} (type={spec['type']}, id={spec.get('model_id')})")
    print(f"Gold: {len(texts)} posts / {len(set(story_ids))} stories  |  num_topics(k)={k}")

    if spec["type"] == "gensim_lda":
        predictor = LDAPredictor(spec)
    elif spec["type"] == "bertopic":
        predictor = BERTopicPredictor(spec, args.bertopic_cluster)
    else:
        sys.exit(f"ERROR: eval_topics.py does not handle type '{spec['type']}'.")

    # ---- ACCURACY on the gold set --------------------------------------------
    preds, gold_secs = predictor.fit_predict(texts, k)
    ari, nmi = clustering_metrics(story_ids, preds)
    frag = fragmentation(story_ids, preds)

    # ---- LATENCY on the full pool --------------------------------------------
    pool_secs = pool_n = None
    pool_capped = False
    if not args.skip_pool and os.path.exists(args.pool):
        pool_texts = load_pool(args.pool, args.pool_limit)
        pool_n = len(pool_texts)
        pool_capped = bool(args.pool_limit and pool_n >= args.pool_limit)
        print(f"  timing fit+transform on pool ({pool_n} docs"
              f"{', capped' if pool_capped else ''}) ...")
        _, pool_secs = predictor.fit_predict(pool_texts, k)
    elif not args.skip_pool:
        print(f"  (pool not found at {args.pool} — skipping latency run; "
              "run scripts/prepare_topics_data.py)", file=sys.stderr)

    # ---- write outputs -------------------------------------------------------
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"topics_{sanitize(spec['name'])}_{ts}"
    per_row_path = os.path.join(results_dir, base + ".csv")
    summary_path = os.path.join(results_dir, base + ".summary.json")

    with open(per_row_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "story_id", "story_label", "predicted_topic", "text"])
        for i, (t, sid, lab, p) in enumerate(zip(texts, story_ids, labels, preds)):
            w.writerow([i, sid, lab, p, t])

    summary = {
        "task": "topics",
        "model": spec["name"], "model_type": spec["type"], "model_id": spec.get("model_id"),
        "embedding_model": spec.get("model_id") if spec["type"] == "bertopic" else None,
        "bertopic_cluster": args.bertopic_cluster if spec["type"] == "bertopic" else None,
        "gold_set": os.path.relpath(args.gold, HERE),
        "n_gold_posts": len(texts), "n_gold_stories": len(set(story_ids)), "num_topics": k,
        "metrics": {"ARI": ari, "NMI": nmi},
        "fragmentation": frag,
        "latency": {
            "gold_fit_transform_s": round(gold_secs, 3),
            "pool_fit_transform_s": round(pool_secs, 3) if pool_secs is not None else None,
            "pool_n_docs": pool_n,
            "pool_capped": pool_capped,
        },
        "note": spec.get("note", ""),
        "placeholder_gold": "topics_gold.csv is PLACEHOLDER — expand before trusting numbers.",
        "timestamp": ts,
        "per_row_csv": os.path.relpath(per_row_path, HERE),
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # ---- console summary -----------------------------------------------------
    print(f"\n  [{spec['name']}]  ARI={ari:.3f}  NMI={nmi:.3f}  purity={frag['cluster_purity']:.3f}")
    print(f"    predicted topics: {frag['n_predicted_topics']}  (gold stories: {frag['n_gold_stories']})")
    print(f"    FRAGMENTED stories (split across >1 topic): {frag['n_fragmented_stories']}/{frag['n_gold_stories']}")
    print(f"    MERGED topics (contain >1 gold story):      {frag['n_merged_topics']}/{frag['n_predicted_topics']}")
    for s, info in frag["per_story"].items():
        flag = "  <-- FRAGMENTED" if info["n_topics"] > 1 else ""
        print(f"      story {s}: {info['n_posts']} posts -> topics {info['predicted_topics']}{flag}")
    print(f"    latency: gold fit+transform={gold_secs:.2f}s"
          + (f" | pool({pool_n})={pool_secs:.2f}s" if pool_secs is not None else ""))
    print(f"    -> {os.path.basename(summary_path)}")


if __name__ == "__main__":
    main()
