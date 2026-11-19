# --- sources ---------------------------------------------------------------

# Broad on purpose: relevant work often lives outside cs.IR (k-means variants in
# cs.LG, graph/geometry algorithms in cs.DS/cs.CG, GPU/SSD systems in cs.DC/cs.AR).
# There is no keyword prefilter; the Claude triage pass decides relevance.
ARXIV_CATEGORIES = [
    "cs.IR",  # information retrieval
    "cs.DB",  # databases
    "cs.DS",  # data structures & algorithms
    "cs.CG",  # computational geometry
    "cs.LG",  # machine learning
    "cs.CL",  # computation & language (embeddings, RAG)
    "cs.DC",  # distributed / parallel computing
    "cs.AR",  # hardware architecture
    "cs.PF",  # performance
]
MAX_ARXIV_RESULTS_PER_CATEGORY = 1500  # safety cap; paginated
ARXIV_PAGE_SIZE = 250
ARXIV_REQUEST_DELAY_S = 3  # arXiv asks for >= 3s between API calls

# arXiv submission time != announcement time (weekends, holidays), so look back a
# few days and rely on the seen-paper ledger to avoid repeats.
LOOKBACK_DAYS = 4
HF_LOOKBACK_DAYS = 2
SEEN_RETENTION_DAYS = 30

ARXIV_API = "https://export.arxiv.org/api/query"
HF_PAPERS_API = "https://huggingface.co/api/daily_papers"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}

# --- claude ----------------------------------------------------------------

TRIAGE_MODEL = "claude-opus-5-5"
TRIAGE_EFFORT = "low"
TRIAGE_BATCH_SIZE = 100  # papers per triage request
TRIAGE_MAX_TOKENS = 16000

DIGEST_MODEL = "claude-opus-5-5"
DIGEST_EFFORT = "high"
DIGEST_MAX_TOKENS = 64000

MIN_TRIAGE_SCORE = 2  # 0-3; papers below this are dropped
MAX_PAPERS_IN_DIGEST = 25

# Message Batches API: 50% off, results usually within an hour (max 24h).
USE_BATCH_API = True
BATCH_POLL_INTERVAL_S = 30
BATCH_TIMEOUT_S = 4 * 60 * 60  # then cancel and finish the rest with direct calls

# Server-side refusal fallback for direct (non-batch) calls. The Batches API does not
# accept it, so refused batch requests are retried directly. Not supported on Haiku 5.5.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MODELS_WITHOUT_FALLBACK = {"claude-haiku-5-5"}

FOCUS_DESCRIPTION = """\
The reader builds vector search engines (think Qdrant, Faiss, Milvus, DiskANN internals).
They care about anything that could change how vectors are indexed, compressed, searched,
or served, including work that never says "vector search".

Directly relevant:
- approximate nearest neighbor search: graph indexes (HNSW, Vamana/DiskANN, NSG), IVF,
  LSH, tree methods; construction, search, updates/deletes, streaming and freshness
- filtered / hybrid / multi-vector / range / multi-tenant search
- vector quantization and compression: PQ, OPQ, scalar/binary, RaBitQ, additive, learned
- vector databases and systems: disk/SSD-resident indexes, tiered memory, distributed and
  serverless search, GPU and accelerator search, SIMD distance kernels
- ANN benchmarks, out-of-distribution queries, recall/latency evaluation methodology

Indirectly relevant - these are the papers a keyword filter misses; score them generously
when the technique plausibly transfers to vector indexing or search:
- clustering at scale (k-means variants, balanced/hierarchical clustering, centroid
  initialization, coresets) - this is how IVF partitions are built
- dimensionality reduction, random projections, Johnson-Lindenstrauss, sketching
- high-dimensional geometry, metric spaces, maximum inner product search, similarity joins
- graph algorithms that apply to proximity graphs (navigability, pruning, routing)
- learned indexes and learned data structures, learned routing / partitioning
- compression or quantization of embeddings, KV caches, or model weights when the method
  works on generic vectors
- systems work on memory hierarchy, caching, prefetching, I/O scheduling for random reads
- embedding models when they change what is stored or searched (Matryoshka, binary,
  sparse/learned-sparse, late interaction, embedding dimension or precision)
- retrieval for RAG and agents only when it involves the retrieval/index layer itself

Calibration examples:
- "GINGER: Spicing Up Graph-Based Vector Search" (graph ANN library, new quantizer,
  compressed proximity graphs) -> 3, core vector search.
- "e-means: outperforming k-means (in one-tenth the epochs)" (a faster mini-batch k-means,
  filed under cs.LG, never mentions vector search) -> 3: k-means builds IVF partitions and
  PQ codebooks, so a 10x faster k-means on embeddings directly speeds up index builds.

Not relevant: RAG papers that only change prompting or generation, LLM applications that
merely use retrieval, generic recommendation models, and IR work about relevance labeling
or user studies with no indexing/search component."""


# --- output schemas ------------------------------------------------------------

TRIAGE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "papers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "score": {
                        "type": "integer",
                        "enum": [0, 1, 2, 3],
                        "description": "0 irrelevant, 1 tangential, 2 transferable to vector search, 3 core vector search",
                    },
                },
                "required": ["id", "score"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["papers"],
    "additionalProperties": False,
}

DIGEST_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "highlights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "reason": {"type": "string", "description": "one sentence on why this is a must-read"},
                },
                "required": ["title", "url", "reason"],
                "additionalProperties": False,
            },
        },
        "clusters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "theme": {"type": "string", "description": "short theme name"},
                    "papers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "url": {"type": "string"},
                                "authors": {"type": "array", "items": {"type": "string"}},
                                "published": {"type": "string", "description": "YYYY-MM-DD"},
                                "source": {"type": "string"},
                                "relevance": {"type": "integer", "enum": [1, 2, 3]},
                                "summary": {"type": "string", "description": "2-3 sentence summary"},
                                "vector_search_angle": {
                                    "type": "string",
                                    "description": "one sentence on how this applies to vector search; essential when the link is indirect",
                                },
                            },
                            "required": [
                                "title", "url", "authors", "published", "source",
                                "relevance", "summary", "vector_search_angle",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["theme", "papers"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["highlights", "clusters"],
    "additionalProperties": False,
}
