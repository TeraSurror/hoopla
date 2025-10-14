#! /usr/bin/env python3

import argparse

from lib.search_utils import BM25_B, BM25_K1
from lib.keyword_search import bm25_command, bm25_idf_command, bm25_tf_command, build_command, idf_command, search_command, tf_command, tf_idf_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the inverted index from the given movies")

    tf_parser = subparsers.add_parser("tf", help="Get the TF for a given term")
    tf_parser.add_argument("document_id", type=int, help="Document id for a document")
    tf_parser.add_argument("term", type=str, help="term for which to get frequency")

    idf_parser = subparsers.add_parser("idf", help="Get the IDF given a term")
    idf_parser.add_argument("term", type=str, help="term for which to get idf")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get the TF-IDF for a given document and term")
    tfidf_parser.add_argument("document_id", type=int, help="Document id for a document")
    tfidf_parser.add_argument("term", type=str, help="term for which to get frequency")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get bm25 score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="term for which to get idf")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get the BM25 TF score for a given document and term")
    bm25_tf_parser.add_argument("document_id", type=int, help="Document id for a document")
    bm25_tf_parser.add_argument("term", type=str, help="term for which to get frequency")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 B parameter")

    bm25_parser = subparsers.add_parser("bm25search", help="Search movies using BM25 scoring")
    bm25_parser.add_argument("query", type=str, help="Search query")
    bm25_parser.add_argument("limit", type=int, nargs="?", default=5, help="Search query")

    
    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            movie_list = search_command(args.query)
            for idx, movie in enumerate(movie_list):
                print(f"{idx + 1}. ({movie["id"]}) {movie['title']}")
        case "bm25search":
            print(f"Searching for: {args.query}")
            movie_list = bm25_command(args.query, args.limit)
            for idx, movie in enumerate(movie_list):
                mov, score = movie
                print(f"{idx + 1}. ({mov["id"]}) {mov['title']} - Score: {score:.2f}")
        case "build":
            print("Building inverted index")
            build_command()
            print("Inverted index built successfully")
        case "tf":
            tf = tf_command(args.document_id, args.term)
            print(f"Term frequency of '{args.term}' in '{args.document_id}': {tf:.2f}")
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of {args.term}': {idf:.2f}")
        case "tfidf":
            tf_idf = tf_idf_command(args.document_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.document_id}': {tf_idf:.2f}")
        case "bm25idf":
            bm25_idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.document_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.document_id}': {bm25tf:.2f}")
        case "bm25":
            bm25 = bm25_command(args.document_id, args.term)
            print(f"BM25 score of '{args.term}' in document '{args.document_id}': {bm25:.2f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()


    