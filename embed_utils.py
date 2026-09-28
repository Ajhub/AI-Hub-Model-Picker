"""
embed_utils.py - embedding + ranking helpers for the AI Hub Model Picker.

Everything here runs locally on CPU. After the first run downloads the small
all-MiniLM-L6-v2 model, no network access is needed.
"""

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent
CATALOG_PATH = BASE_DIR / "catalog.json"
EMBEDDINGS_PATH = BASE_DIR / "embeddings.npy"
INDEX_PATH = BASE_DIR / "embeddings_index.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def get_model(device=None):
    """Load the sentence-transformer once per device and reuse it.

    device=None  -> let the library pick its default backend
    device="cpu" -> force CPU-only inference
    """
    # Imported here so the rest of the module (and its tests) stay importable
    # even if the heavy dependency is not installed yet.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, device=device)


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------
def load_catalog():
    """Read catalog.json and return the list of model entries."""
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _catalog_text(entry):
    """Text that represents one model for embedding."""
    return f"{entry['name']}. {entry['category']}. {entry['description']}"


def embed_catalog(force=False):
    """Embed every model description once and cache the result to disk.

    Writes:
      embeddings.npy         - array of shape (n_models, dim)
      embeddings_index.json  - list mapping row i -> catalog entry name

    If the cache already exists (and force is False) it is reused.
    """
    catalog = load_catalog()

    if not force and EMBEDDINGS_PATH.exists() and INDEX_PATH.exists():
        cached_names = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        if cached_names == [m["name"] for m in catalog]:
            return np.load(EMBEDDINGS_PATH)

    model = get_model()
    texts = [_catalog_text(m) for m in catalog]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype=np.float32)

    np.save(EMBEDDINGS_PATH, vectors)
    INDEX_PATH.write_text(
        json.dumps([m["name"] for m in catalog], indent=2), encoding="utf-8"
    )
    return vectors


# ---------------------------------------------------------------------------
# Query embedding + ranking
# ---------------------------------------------------------------------------
def embed_query(text, device=None):
    """Embed a single user query string into a (1, dim) vector."""
    model = get_model(device)
    vec = model.encode([text], normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vec, dtype=np.float32)


def rank_matches(query_vec, catalog_vecs, top_k=5):
    """Rank catalog rows by cosine similarity to the query.

    Returns (indices, scores): both arrays are sorted best-first, and
    indices point into the catalog list / rows of catalog_vecs.
    """
    scores = cosine_similarity(query_vec, catalog_vecs)[0]
    top_k = min(top_k, len(scores))
    order = np.argsort(scores)[::-1][:top_k]
    return order, scores[order]


# ---------------------------------------------------------------------------
# Timing helper (used by the CPU-vs-NPU expander in app.py)
# ---------------------------------------------------------------------------
def time_embedding(text, device=None, runs=20):
    """Average time in milliseconds to embed `text` on the given device."""
    model = get_model(device)
    model.encode([text], normalize_embeddings=True, show_progress_bar=False)  # warm-up

    start = time.perf_counter()
    for _ in range(runs):
        model.encode([text], normalize_embeddings=True, show_progress_bar=False)
    elapsed = time.perf_counter() - start
    return (elapsed / runs) * 1000.0


if __name__ == "__main__":
    vecs = embed_catalog(force=True)
    print(f"Embedded {vecs.shape[0]} models -> {EMBEDDINGS_PATH.name} (dim={vecs.shape[1]})")
