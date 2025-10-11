import string
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stop_words
from nltk.stem import PorterStemmer


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()

    movie_list = []

    preprocessed_query = tokenize_text(query)
    for movie in movies:
        preprocessed_movie_title = tokenize_text(movie["title"])


        if check_token_match(preprocessed_query, preprocessed_movie_title):
            movie_list.append(movie)

        if len(movie_list) == limit:
            break

    return movie_list

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

    stopwords = load_stop_words()

    for token in tokens:
        if token and token not in stopwords:
            valid_tokens.append(stem_word(text))

    return valid_tokens

def stem_word(text: str) -> str:
    stemmer = PorterStemmer()
    
    return stemmer.stem(text)

def preprocess_text(text: str) -> str:
    text =  text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    return text
