from calendar import c
import chunk
import json
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import CACHE_PATH, CHUNK_EMBEDDING_PATH, CHUNK_METADATA_PATH, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP_SIZE, DEFAULT_SEARCH_LIMIT, DOCUMENT_PREVIEW_LENGTH, EMBEDDING_PATH, format_search_result, load_movies

class SemanticSearch:
    def __init__(self, model="allMiniLM-L6-v2"):
        self.model = SentenceTransformer(model)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("Input text is empty")

        return self.model.encode([text])[0]

    def build_embeddings(self, documents):
        self.documents = documents
        movie_text_list = []

        for document in documents:
            self.document_map[document["id"]] = document
            movie_data = f"{document["title"]}: {document["description"]}"
            movie_text_list.append(movie_data)

        self.embeddings = self.model.encode(movie_text_list, show_progress_bar=True)

        os.makedirs(EMBEDDING_PATH, exist_ok=True)
        np.save(EMBEDDING_PATH, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents

        for document in documents:
            self.document_map[document["id"]] = document

        if os.path.exists(EMBEDDING_PATH):
            self.embeddings = np.load(EMBEDDING_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None or len(self.embeddings) == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first")

        if self.documents is None or len(self.documents) == 0:
            raise ValueError("No documents found. call `load_or_create_embeddings` first")

        query_embeddings = self.generate_embedding(query)

        similarities = []
        for idx, doc_embedding in enumerate(self.embeddings):
            cs = cosine_similarity(query_embeddings, doc_embedding)
            similarities.append((cs, self.documents[idx]))

        similarities.sort(key=lambda x: x[0], reverse=True)

        result = []
        for similarity, movie in similarities[:limit]:
            result.append({
                "score": similarity,
                "title": movie["title"],
                "description": movie["description"]
            })

        return result

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}

        all_chunks = []
        all_chunk_metadata = []
        for idx, document in enumerate(self.documents):
            doc_id = document["id"]
            description = document.get("description", "")

            self.document_map[doc_id] = document

            if not description or not description.strip():
                continue

            chunks = semantic_chunking(description, 4, 1)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_chunk_metadata.append({
                    "movie_idx": idx,
                    "chunk_idx": i,
                    "total_chunks": len(chunks)
                })
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = all_chunk_metadata

        os.makedirs(os.path.dirname(CHUNK_EMBEDDING_PATH), exist_ok=True)
        np.save(CHUNK_EMBEDDING_PATH, self.chunk_embeddings)

        with open(CHUNK_METADATA_PATH, 'w') as f:
            json.dump({"chunks": all_chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}

        for document in self.documents:
            doc_id = document["id"]

            self.document_map[doc_id] = document

            if os.path.exists(CHUNK_EMBEDDING_PATH) and os.path.exists(CHUNK_METADATA_PATH):
                self.chunk_embeddings = np.load(CHUNK_EMBEDDING_PATH, allow_pickle=True)
                with open(CHUNK_METADATA_PATH, 'r') as f:
                    data = json.load(f)
                    self.chunk_metadata = data["chunks"]
                
                return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query, limit):
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError("No chunk embeddings are loaded. Call `load_or_create_chunk_embeddings` first.")

        embeddings = self.generate_embedding(query)
        chunk_scores = []

        for idx, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(embeddings, chunk_embedding)
            chunk_scores.append({
                "chunk_idx": idx,
                "movie_idx": self.chunk_metadata[idx]["movie_idx"],
                "score": similarity
            })

        movie_scores = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            if (
                movie_idx not in movie_scores
                or chunk_score["score"] > movie_scores[movie_idx]
            ):
                movie_scores[movie_idx] = chunk_score["score"]

        sorted_movies = sorted(movie_scores.items(), key=lambda x:x[1], reverse=True)

        results = []
        for movie_idx, score in sorted_movies[:limit]:
            doc = self.documents[movie_idx]
            results.append(
                format_search_result(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"][:DOCUMENT_PREVIEW_LENGTH],
                    score=score
                )
            )
        
        return results


def embed_chunks():
    chunkedSemanticSearcher = ChunkedSemanticSearch()
    movie_list = load_movies()
    embeddings = chunkedSemanticSearcher.load_or_create_chunk_embeddings(movie_list)
    print(f"Generated {len(embeddings)} chunked embeddings")


def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    movie_list = load_movies()
    semantic_search = SemanticSearch()
    embeddings = semantic_search.load_or_create_embeddings(movie_list)

    print(f"Number of docs: {len(movie_list)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def search_movies(query, limit):
    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(load_movies())
    movie_list = semantic_search.search(query, limit)

    for idx, movie in enumerate(movie_list):
        print(f"{idx + 1}. {movie["title"]} (score: {movie["score"]:.4f})")
        print(f"   {movie["description"][:100]}...")

def chunk_text(text, chunk_size = DEFAULT_CHUNK_SIZE, overlap_size = DEFAULT_OVERLAP_SIZE):
    chunks = fixed_size_chunking(text, chunk_size, overlap_size)
    print(f"Chunking {len(text)} characters")

    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")

def fixed_size_chunking(text, chunk_size = DEFAULT_CHUNK_SIZE, overlap_size = DEFAULT_OVERLAP_SIZE):
    words = text.split(" ")
    chunks = []

    i = 0
    while i < len(words) - overlap_size:
        chunk_words = words[i:i+chunk_size]
        chunks.append(" ".join(chunk_words))
        i += (chunk_size - overlap_size)

    return chunks

def semantic_chunk_text(text, chunk_size = DEFAULT_CHUNK_SIZE, overlap_size = DEFAULT_OVERLAP_SIZE):
    chunks = semantic_chunking(text, chunk_size, overlap_size)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")

def semantic_chunking(text, chunk_size = DEFAULT_CHUNK_SIZE, overlap_size = DEFAULT_OVERLAP_SIZE):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    
    i = 0
    while i < len(sentences) - overlap_size:
        chunk_sentences = sentences[i:i+chunk_size]
        chunks.append(" ".join(chunk_sentences))
        i += (chunk_size - overlap_size)

    return chunks

def search_chunked_command(query, limit = DEFAULT_SEARCH_LIMIT):
    movie_list = load_movies()
    searcher = ChunkedSemanticSearch()
    searcher.load_or_create_chunk_embeddings(movie_list)
    results = searcher.search_chunks(query, limit)
    return {"query": query, "results": results}