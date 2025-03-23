# ArXiv Research Assistant v0.1.0

## Overview
ArXiv Research Assistant is a powerful tool that searches arXiv for papers on a given topic, ranks them by relevance using semantic similarity, and provides summaries. This release marks the first stable version of the tool.

## Features
- **Intelligent Paper Search**: Searches arXiv for papers based on user queries
- **Semantic Ranking**: Ranks papers by the relevance of their abstracts to the query using embeddings
- **Adaptive Model Loading**: Falls back to simpler models if advanced models aren't available
- **Paper Summarization**: Generates summaries and answers questions about the research papers
- **Robust Error Handling**: Gracefully handles failures at various stages of the pipeline

## Technical Details
- Uses `sentence-transformers` for embedding text and semantic similarity
- Leverages `langchain` for creating retrieval chains
- Employs `transformers` for text generation with fallback options
- Utilizes `faiss-cpu` for efficient similarity search
- Implements a tiered approach to model loading to ensure functionality across different environments

## Installation

### Prerequisites
- Python 3.8+
- Poetry for dependency management

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/MeherBhaskar/arxiv_search_helper.git
   cd arxiv_search_helper
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Configure your Hugging Face token in `config.py` or as an environment variable.

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

## Known Issues
- May experience slow performance on systems with limited memory
- Some language models might fail to load on certain environments
- API rate limiting from arXiv can affect search results

## Contributors
- Meher Bhaskar (@MeherBhaskar)

## License
MIT License
