# Web Crawler Search Engine

A lightweight Python web crawler and search engine with TF-IDF ranked search via CLI.

## Features

- Crawl websites breadth‑first with politeness delay and domain scoping
- Build an inverted index stored in SQLite for persistence
- Rank results using TF‑IDF weighting
- Interactive search REPL with explanations and statistics
- Single‑query mode for scripting
- Reset and stats commands for index management

## Getting started

Prerequisites: Python 3 and pip.

Install dependencies:

```bash
pip install -r requirements.txt
```

To crawl a website and build the index:

```bash
python main.py crawl --url <SEED_URL> --pages <MAX_PAGES> --depth <DEPTH>
```

Example:

```bash
python main.py crawl --url https://en.wikipedia.org/wiki/Python_(programming_language) --pages 30 --depth 2
```

To search interactively:

```bash
python main.py search
```

For a one‑shot query:

```bash
python main.py search --query "list comprehension"
```

To show index statistics:

```bash
python main.py stats
```

To reset the index:

```bash
python main.py reset
```

## Project structure

- `main.py` – CLI entry point with subcommands (crawl, search, reset, stats)
- `crawler.py` – Breadth‑first web crawler that extracts text and links
- `indexer.py` – SQLite‑backed inverted index builder and reader
- `ranker.py` – TF‑IDF ranking engine with explanation helper
- `search.py` – Interactive REPL and single‑query interface
- `utils.py` – Text preprocessing (tokenization, stopwords, TF/IDF helpers)

## License

Released under the MIT License (see LICENSE).
