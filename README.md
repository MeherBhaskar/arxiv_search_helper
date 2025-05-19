
# ArXiv Research Assistant 🔬📚

A powerful AI-powered research tool that helps you navigate the vast world of scientific literature on arXiv. Using advanced natural language processing and semantic search capabilities, it finds, ranks, and summarizes research papers that are most relevant to your query.

## Key Features

### 🎯 Intelligent Paper Discovery
- Performs targeted searches on arXiv using your research queries
- Smart ranking system using semantic similarity to identify the most relevant papers
- Automatically extracts and processes paper abstracts and metadata

### 🧠 Advanced AI Processing
- Uses state-of-the-art embedding models for semantic understanding of research papers
- Implements vector-based similarity search for precise paper ranking
- Generates concise, informative summaries of complex research papers
- Answers questions about the papers using advanced language models

### 💪 Robust Architecture
- Adaptive model loading system that works across different computing environments
- Graceful fallback to simpler models when needed, ensuring consistent functionality
- Efficient vector storage and retrieval using FAISS
- Comprehensive error handling for reliable operation

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Poetry for dependency management
- (Optional) Hugging Face API token for access to more advanced models

### Quick Installation

1. Clone the repository:
```bash
git clone https://github.com/MeherBhaskar/arxiv_search_helper.git
cd arxiv_search_helper
```

2. Set up the environment:
```bash
poetry install
```

3. (Optional) Configure your Hugging Face token:
   - Add your token to `config.py` or
   - Set as environment variable: `export HUGGINGFACE_TOKEN=your_token_here`

### Usage

1. Start the assistant:
```bash
poetry run python arxiv_search.py
```

2. Enter your research query when prompted
   - Try specific queries like "quantum computing applications in finance"
   - Use natural language questions
   - Type 'exit' to quit

The assistant will:
- Search for relevant papers
- Rank them by semantic similarity
- Generate summaries
- Provide answers based on the papers' content

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

## 🔧 How It Works

### Pipeline Overview
1. **Paper Discovery**: Searches arXiv's extensive database using your research query
2. **Text Processing**: Extracts and processes paper titles, abstracts, and metadata
3. **Semantic Analysis**: Uses AI embeddings to understand the content of each paper
4. **Smart Ranking**: Ranks papers by semantic similarity to your query
5. **Content Generation**: Generates summaries and answers using language models

### 🛠️ Technical Stack

#### Core Components
- **arxiv**: Interface with arXiv's research database
- **sentence-transformers**: State-of-the-art text embedding models
- **langchain**: Advanced NLP pipelines and retrieval systems
- **transformers**: Cutting-edge language models for text generation
- **faiss-cpu**: High-performance vector similarity search

#### Key Features
- Semantic similarity scoring using cosine similarity
- Adaptive model loading with fallback options
- Efficient vector storage and retrieval
- Robust error handling and recovery

## 📝 Contributing

Contributions are welcome! Feel free to submit issues and enhancement requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details
