# Hoopla

A Python CLI tool for searching movies using keyword-based text matching with natural language processing features.

## Overview

Hoopla is a command-line application that allows you to search through a collection of movies using natural language queries. It implements text preprocessing, tokenization, stemming, and stopword filtering to provide relevant movie search results.

## Features

- **Keyword-based search**: Search movies by title using natural language queries
- **Text preprocessing**: Automatic lowercase conversion and punctuation removal
- **Stopword filtering**: Removes common words to improve search relevance
- **Stemming**: Uses Porter Stemmer to match word variations
- **Configurable results**: Limit the number of search results returned
- **CLI interface**: Easy-to-use command-line interface

## Installation

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd hoopla
```

2. Install dependencies using uv:
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

## Usage

### Basic Search

Search for movies using the `search` command:

```bash
python cli/keyword_search_cli.py search "action movie"
```

### Example Queries

```bash
# Search for action movies
python cli/keyword_search_cli.py search "action"

# Search for romantic films
python cli/keyword_search_cli.py search "romance"

# Search for specific actors or directors
python cli/keyword_search_cli.py search "spielberg"

# Search for movie genres
python cli/keyword_search_cli.py search "comedy"
```

### Output Format

The search results are displayed as a numbered list:

```
Searching for: action
1. Kaakha..Kaakha: The Police
2. Grandview, U.S.A.
3. Now You Know
4. Rumpelstilzchen
5. [Additional results...]
```

## How It Works

The search algorithm performs the following steps:

1. **Text Preprocessing**: Converts input to lowercase and removes punctuation
2. **Tokenization**: Splits the query into individual words
3. **Stopword Removal**: Filters out common words (a, the, and, etc.)
4. **Stemming**: Reduces words to their root form using Porter Stemmer
5. **Matching**: Compares processed query tokens with movie titles
6. **Results**: Returns movies where at least one token matches

## Project Structure

```
hoopla/
├── cli/
│   ├── keyword_search_cli.py    # Main CLI entry point
│   └── lib/
│       ├── keyword_search.py    # Core search functionality
│       └── search_utils.py      # Utility functions
├── data/
│   ├── movies.json              # Movie database
│   └── stopwords.txt            # Stopwords list
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## Dependencies

- **nltk (3.9.1)**: Natural Language Toolkit for text processing and stemming

## Data

The project includes:
- A JSON database of movies with titles and descriptions
- A stopwords file for filtering common words during search

## Development

### Running Tests

To run the application in development mode:

```bash
python cli/keyword_search_cli.py search "your query here"
```

### Adding New Movies

To add new movies to the database, edit the `data/movies.json` file following the existing format:

```json
{
  "movies": [
    {
      "id": 1,
      "title": "Movie Title",
      "description": "Movie description..."
    }
  ]
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test your changes
5. Submit a pull request

## License

This project is open source. Please check the license file for more details.

## Future Enhancements

- Implement BM25 ranking algorithm for better search relevance
- Add support for searching movie descriptions
- Implement fuzzy matching for typos
- Add configuration options for search parameters
- Support for multiple search modes (exact match, partial match, etc.)
