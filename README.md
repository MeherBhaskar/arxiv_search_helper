
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

## Example
```
ArXiv Research Assistant
------------------------
This tool searches arXiv for papers on your topic and provides summaries.
The papers are ranked by the relevance of their abstracts to your query.
------------------------

Enter your research query (or 'exit' to quit): quantum computing applications

Searching for papers on arXiv...
Found 20 papers. Extracting text...
Creating vector store...
Trying to load the embedding model...
Ranking papers by abstract relevance...

Paper Ranking by Abstract Relevance:
1. Quantum Computing Applications in Financial Services (Score: 0.8765)
2. Recent Advances in Quantum Computing for Machine Learning (Score: 0.8432)
...

Selected top 10 most relevant papers based on abstract content.
Setting up language model...
Loading language model: facebook/opt-125m...
Successfully loaded facebook/opt-125m
Creating retrieval chain...
Generating answer...

Query: quantum computing applications

Answer: Quantum computing applications span various fields including finance, cryptography,
optimization, and machine learning. In finance, quantum algorithms can improve portfolio
optimization and risk assessment. For cryptography, quantum computers pose both threats to
current encryption methods and opportunities for quantum-secure protocols. In optimization,
quantum approaches can solve complex problems more efficiently than classical methods.
Machine learning benefits from quantum computing through faster training and more complex
model capabilities. Recent research has focused on near-term applications using NISQ devices,
showing promising results despite current hardware limitations.

Top Relevant Papers:
1. Quantum Computing Applications in Financial Services (https://arxiv.org/abs/2201.12345)
   Authors: Jane Smith, John Doe
   Published: 2022-01-15

2. Recent Advances in Quantum Computing for Machine Learning (https://arxiv.org/abs/2202.54321)
   Authors: Alice Johnson, Bob Williams
   Published: 2022-02-20
...
```

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
