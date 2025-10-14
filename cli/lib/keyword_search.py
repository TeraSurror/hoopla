from collections import Counter
import math
import os
import pickle
import string

from nltk import defaultdict

from .search_utils import BM25_B, BM25_K1, CACHE_PATH, DEFAULT_SEARCH_LIMIT, load_movies, load_stop_words
from nltk.stem import PorterStemmer

class InvertedIndex:

    def __init__(self) -> None:
        # a dictionary mapping tokens (strings) to sets of document ids (integers)
        self.index = defaultdict(set)

        # a dictionary mapping document IDs to their full document objects
        self.docmap: dict[int, dict] = {}

        # Dict mapping document and tf of all tokens in the document
        self.term_frequencies: dict[int, dict] = {}

        # Dict mapping document and token length
        self.doc_lengths: dict[int, int] = {}

        self.index_path = os.path.join(CACHE_PATH, "index.pkl")
        self.docmap_path = os.path.join(CACHE_PATH, "docmap.pkl")
        self.doc_lengths_path = os.path.join(CACHE_PATH, "doc_lengths.pkl")
        self.term_frequencies_path = os.path.join(CACHE_PATH, "term_frequencies.pkl")

    def get_documents(self, term: str) -> list[int]: 
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: str, term: str) -> int:
        tokenized_term = tokenize_text(term)[0]

        return self.term_frequencies[doc_id].get(tokenized_term, 0)

    def get_idf(self, term: str) -> float:
        tokenized_term = tokenize_text(term)[0]

        return math.log((len(self.docmap) + 1) / (len(self.index[tokenized_term]) + 1))

    def get_tf_idf(self, doc_id: str, term: str) -> int:
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def get_bm25_idf(self, term: str) -> float:
        tokenized_term = tokenize_text(term)[0]
        doc_len = len(self.docmap)
        doc_freq = len(self.index[tokenized_term])

        return math.log((doc_len - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self._get_avg_doc_length()

        # Length normalization factor
        length_norm = 1 - b + b * (doc_length / avg_doc_length)

        # Apply to term frequency
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)

        return tf_component

    def get_bm25(self, doc_id: int, term: str) -> float:
        bm25_idf = self.get_bm25_idf(term)
        bm25_tf = self.get_bm25_tf(doc_id, term)

        return bm25_idf * bm25_tf

    def bm25_search(self, query: str, limit) -> list[int]:
        tokenized_query = tokenize_text(query)
        scores = {}
        result = []

        for doc_id in self.docmap.keys():
            bm25 = 0.0
            for token in set(tokenized_query):
                bm25 += self.get_bm25(doc_id, token)
            scores[doc_id] = bm25
            result.append((doc_id, bm25))

        result = sorted(result, key=lambda x:x[1], reverse=True)
        result = result[:limit]

        return result

    def build(self) -> None:
        movie_list = load_movies()
        for movie in movie_list:
            doc_id = movie["id"]
            doc_text = f"{movie["title"]} {movie["description"]}"
            self.docmap[doc_id] = movie
            self._add_document(doc_id, doc_text)
        
    def save(self) -> None:
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, 'wb') as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, 'wb') as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, 'wb') as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        with open(self.index_path, 'rb') as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, 'rb') as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, 'rb') as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, 'rb') as f:
            self.doc_lengths = pickle.load(f)

    def _add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id] = Counter(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def _get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0

        result = 0.0
        for doc_length in self.doc_lengths.values():
            result += doc_length
        
        return result / len(self.doc_lengths)

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()

def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()

    return idx.get_tf(doc_id, term)

def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()

    return idx.get_idf(term)
    
def tf_idf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()

    return idx.get_tf_idf(doc_id, term)

def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()

    return idx.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    idx = InvertedIndex()
    idx.load()

    return idx.get_bm25_tf(doc_id, term, k1)

def bm25_command(query: str, limit: int) -> list[dict]:
    idx = InvertedIndex()
    idx.load()

    return [(idx.docmap[i], score) for i, score in idx.bm25_search(query, limit)]

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()

    query_tokens = tokenize_text(query)
    seen, results = set(), []
    for query_token in query_tokens:
        for doc_id in idx.get_documents(query_token):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = idx.docmap[doc_id]
            if not doc:
                continue
            results.append(doc)
            if len(results) >= limit:
                return results

def check_token_match(tokens1: list[str], tokens2: list[str]) -> bool:
    for t1 in tokens1:
        for t2 in tokens2:
            if t1 in t2:
                return True

    return False

def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stop_words = load_stop_words()
    filtered_words = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    return text
