import os
import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import EMBEDDING_PATH, load_movies

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
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
