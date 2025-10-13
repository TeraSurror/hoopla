#! /usr/bin/env python3

import argparse

from lib.keyword_search import build_command, idf_command, search_command, tf_command, tf_idf_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("build", help="Build the inverted index from the given movies")

    search_parser = subparsers.add_parser("tf", help="Get the term frequency")
    search_parser.add_argument("document_id", type=int, help="Document id for a document")
    search_parser.add_argument("term", type=str, help="term for which to get frequency")

    search_parser = subparsers.add_parser("idf", help="Get the inverse document frequency")
    search_parser.add_argument("term", type=str, help="term for which to get idf")

    search_parser = subparsers.add_parser("tfidf", help="Get the TF-IDF")
    search_parser.add_argument("document_id", type=int, help="Document id for a document")
    search_parser.add_argument("term", type=str, help="term for which to get frequency")
    
    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            movie_list = search_command(args.query)
            for idx, movie in enumerate(movie_list):
                print(f"{idx + 1}. ({movie["id"]}) {movie['title']}")
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
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()


    