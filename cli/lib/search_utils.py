import json
import os

DEFAULT_SEARCH_LIMIT = 5
DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAP_SIZE = 0
DOCUMENT_PREVIEW_LENGTH = 100
SCORE_PRECISION = 3

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_FILE_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
EMBEDDING_PATH = os.path.join(CACHE_PATH, "movie_embeddings.npy")
CHUNK_EMBEDDING_PATH = os.path.join(CACHE_PATH, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_PATH, "chunk_metadata.json")

BM25_K1 = 1.5
BM25_B = 0.75

def load_movies() -> list[dict]:
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)

    return data["movies"]

def load_stop_words() -> list[str]:
    with open(STOPWORDS_FILE_PATH, 'r') as f:
        return f.read().splitlines()

def format_search_result(doc_id, title, document, score, **metadata):
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }


