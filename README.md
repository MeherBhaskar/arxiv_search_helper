# ArXiv Research Assistant

A tool that searches arXiv for papers on a given topic, ranks them by relevance, and provides summaries.

## Features

- Searches arXiv for papers based on user queries
- Ranks papers by the relevance of their abstracts to the query
- Uses embeddings to find the most semantically similar papers
- Generates summaries and answers questions about the research papers
- Falls back to simpler models if advanced models aren't available

## Requirements

- Python 3.8+
- Poetry for dependency management

## Installation

1. Clone this repository:
```bash
git clone https://github.com/MeherBhaskar/arxiv_search_helper.git
cd arxiv_search_helper
```

2. Install dependencies using Poetry:
```bash
poetry install
```

## Usage

Run the main script:
```bash
poetry run python arxiv_search.py
```

Enter your research query when prompted, or type 'exit' to quit.

## How It Works

1. The tool searches arXiv for papers related to your query
2. It extracts the abstracts and titles from the papers
3. It ranks the papers by the relevance of their abstracts to your query
4. It selects the top most relevant papers
5. It uses a language model to generate an answer based on these papers

## Dependencies

- arxiv: For searching and retrieving papers from arXiv
- sentence-transformers: For embedding text
- langchain: For creating retrieval chains
- transformers: For text generation
- faiss-cpu: For efficient similarity search
