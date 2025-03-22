import os
import warnings

import arxiv

warnings.filterwarnings("ignore")
import numpy as np
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LangChainFAISS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from config import HF_TOKEN, MAX_FINAL_RESULTS, MAX_INIT_RESULTS

# Set Hugging Face token if available
# You can set this in your environment variables or directly here
os.environ["HUGGINGFACE_TOKEN"] = HF_TOKEN


# 1. Search for research papers on arXiv based on a query
def search_arxiv(query, max_results=15):  # Increased to get more initial papers
    search = arxiv.Search(
        query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    )
    results = []
    for result in search.results():
        results.append(
            {
                "title": result.title,
                "summary": result.summary,
                "url": result.entry_id,
                "authors": ", ".join(author.name for author in result.authors),
                "published": result.published.strftime("%Y-%m-%d")
                if hasattr(result, "published")
                else "Unknown",
            }
        )
    return results


# 2. Extract text (here, we use the abstracts directly)
def extract_texts_from_results(results):
    texts = []
    for paper in results:
        # Combine title and abstract for better context
        combined_text = f"Title: {paper['title']}\nAbstract: {paper['summary']}"
        texts.append(combined_text)
    return texts


# 3. Build a vector store using embeddings
def create_vectorstore(texts):
    try:
        # First try with the original model
        print("Trying to load the embedding model...")
        hf_embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vectorstore = LangChainFAISS.from_texts(texts, hf_embeddings)
        return vectorstore, hf_embeddings
    except Exception as e:
        print(f"Error loading the original model: {e}")
        print("Falling back to a local SentenceTransformer model...")
        try:
            # Try to download and use a different model that doesn't require authentication
            model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

            # Create a custom embedding function
            def embed_documents(documents):
                return model.encode(documents).tolist()

            def embed_query(query):
                return model.encode(query).tolist()

            # Create a custom embeddings object
            class CustomEmbeddings:
                def __init__(self, embed_documents_func, embed_query_func):
                    self.embed_documents = embed_documents_func
                    self.embed_query = embed_query_func

                def embed_query(self, text):
                    return embed_query(text)

                def embed_documents(self, documents):
                    return embed_documents(documents)

            custom_embeddings = CustomEmbeddings(embed_documents, embed_query)
            vectorstore = LangChainFAISS.from_texts(texts, custom_embeddings)
            return vectorstore, custom_embeddings
        except Exception as fallback_error:
            print(f"Error with fallback model: {fallback_error}")
            raise


# 4. Set up a smaller, more accessible LLM
def create_llm():
    # List of models to try in order of preference - using smaller models first
    models_to_try = [
        "facebook/opt-125m",
        "distilgpt2",  # Smallest model, try first
        "gpt2-medium",
    ]

    import torch

    # Double-check that we're using CPU
    if torch.cuda.is_available():
        print("CUDA is available but we're forcing CPU usage for compatibility")

    # Get Hugging Face token from environment
    HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", None)

    # Suppress specific warnings about truncation
    warnings.filterwarnings("ignore", message=".*Truncation was not explicitly activated.*")
    warnings.filterwarnings("ignore", message=".*Setting `pad_token_id` to `eos_token_id`.*")
    warnings.filterwarnings("ignore", message=".*Input length of input_ids is.*")

    for model_name in models_to_try:
        try:
            print(f"Loading language model: {model_name}...")

            # First approach: Try loading with explicit CPU mapping
            try:
                # Explicitly specify CPU device
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, cache_dir="./hf_cache", use_auth_token=HF_TOKEN
                )

                # Set pad token if it doesn't exist
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                # Try loading with CPU device map but catch specific accelerate-related errors
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        cache_dir="./hf_cache",
                        use_auth_token=HF_TOKEN,
                        device_map="cpu",  # Force CPU
                        torch_dtype=torch.float32,  # Use float32 instead of float16 for CPU compatibility
                    )

                    # If we get here, the model loaded with device_map successfully
                    # Create pipeline without truncation or max_length
                    hf_pipeline = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        do_sample=True,
                        temperature=0.7
                        # No device parameter here since accelerate is handling it
                    )

                except Exception as device_map_error:
                    # If we get an error about accelerate, try without device_map
                    print(f"Device map approach failed: {device_map_error}")
                    print("Trying without device_map...")

                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        cache_dir="./hf_cache",
                        use_auth_token=HF_TOKEN,
                        torch_dtype=torch.float32,  # Use float32 instead of float16 for CPU compatibility
                    )

                    # Move model to CPU explicitly after loading
                    model = model.to("cpu")

                    # Create pipeline without truncation or max_length
                    hf_pipeline = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        do_sample=True,
                        temperature=0.7,
                        device=-1,  # -1 means CPU
                    )

                # Define a simple function that LangChain can use
                def generate_text(prompt):
                    try:
                        result = hf_pipeline(prompt, max_new_tokens=10000)
                        if isinstance(result, list) and len(result) > 0:
                            generated_text = result[0].get("generated_text", "")
                            # If the generated text starts with the prompt, remove it
                            if generated_text.startswith(prompt):
                                generated_text = generated_text[len(prompt) :].strip()
                            return generated_text
                        return "No result generated."
                    except Exception as e:
                        print(f"Generation error: {e}")
                        return f"Based on the research papers, several relevant studies were found."

                # Create a simple LLM that just calls our function
                from typing import Any, List, Mapping, Optional

                from langchain.llms.base import LLM

                class SimpleLLM(LLM):
                    generation_function: Any

                    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
                        return self.generation_function(prompt)

                    @property
                    def _identifying_params(self) -> Mapping[str, Any]:
                        return {"name": "SimpleLLM"}

                    @property
                    def _llm_type(self) -> str:
                        return "simple"

                print(f"Successfully loaded {model_name}")
                return SimpleLLM(generation_function=generate_text)

            except Exception as e:
                print(f"First approach failed: {e}")
                print("Trying alternative loading method...")

                # Second approach: Try loading with default parameters and move to CPU after
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, cache_dir="./hf_cache", use_auth_token=HF_TOKEN
                )

                # Set pad token if it doesn't exist
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                model = AutoModelForCausalLM.from_pretrained(
                    model_name, cache_dir="./hf_cache", use_auth_token=HF_TOKEN
                )

                # Create pipeline without truncation or max_length
                hf_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    do_sample=True,
                    temperature=0.7,
                )

                # Define a simple function that LangChain can use
                def generate_text(prompt):
                    try:
                        result = hf_pipeline(prompt, max_new_tokens=10000)
                        if isinstance(result, list) and len(result) > 0:
                            generated_text = result[0].get("generated_text", "")
                            # If the generated text starts with the prompt, remove it
                            if generated_text.startswith(prompt):
                                generated_text = generated_text[len(prompt) :].strip()
                            return generated_text
                        return "No result generated."
                    except Exception as e:
                        print(f"Generation error: {e}")
                        return f"Based on the research papers, several relevant studies were found."

                # Create a simple LLM that just calls our function
                from typing import Any, List, Mapping, Optional

                from langchain.llms.base import LLM

                class SimpleLLM(LLM):
                    generation_function: Any

                    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
                        return self.generation_function(prompt)

                    @property
                    def _identifying_params(self) -> Mapping[str, Any]:
                        return {"name": "SimpleLLM"}

                    @property
                    def _llm_type(self) -> str:
                        return "simple"

                print(f"Successfully loaded {model_name} with default parameters")
                return SimpleLLM(generation_function=generate_text)

        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            print("Trying next model...")

    # If all models fail, create a simple text generator using the LLM interface
    print("All language models failed. Using a simple text generator...")

    from typing import Any, List, Mapping, Optional

    from langchain.llms.base import LLM

    class FallbackLLM(LLM):
        def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
            return (
                f"Based on the research papers, the query about '{prompt}' relates to several scientific works. "
                f"Please review the papers listed below for detailed information on this topic."
            )

        @property
        def _identifying_params(self) -> Mapping[str, Any]:
            return {"name": "FallbackLLM"}

        @property
        def _llm_type(self) -> str:
            return "fallback"

    return FallbackLLM()


# 5. Create a RetrievalQA chain using LangChain
def create_retrieval_chain(vectorstore, llm):
    try:
        # Create the chain with return_source_documents=True to get both the answer and sources
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True,
        )
        return qa_chain
    except Exception as e:
        print(f"Error creating standard retrieval chain: {e}")
        print("Trying alternative chain setup...")

        try:
            # Try an alternative setup with explicit output keys
            from langchain.chains import LLMChain
            from langchain.prompts import PromptTemplate

            template = """
            Answer the question based on the context below.

            Context:
            {context}

            Question: {question}

            Answer:"""

            prompt = PromptTemplate(template=template, input_variables=["context", "question"])

            llm_chain = LLMChain(llm=llm, prompt=prompt)

            class CustomRetrievalChain:
                def __init__(self, llm_chain, retriever):
                    self.llm_chain = llm_chain
                    self.retriever = retriever

                def invoke(self, query):
                    # Get documents from retriever
                    docs = self.retriever.get_relevant_documents(query)
                    # Join document texts
                    context = "\n\n".join([doc.page_content for doc in docs])
                    # Run the LLM chain
                    result = self.llm_chain.run(context=context, question=query)
                    # Return in the expected format
                    return {"result": result, "source_documents": docs}

            return CustomRetrievalChain(llm_chain, vectorstore.as_retriever())

        except Exception as alt_error:
            print(f"Alternative chain setup failed: {alt_error}")

            # Create a very simple fallback chain
            class SimpleQAChain:
                def __init__(self, retriever):
                    self.retriever = retriever

                def invoke(self, query):
                    docs = self.retriever.get_relevant_documents(query)
                    result = "Please review the research papers listed below for information on this topic."
                    return {"result": result, "source_documents": docs}

                # For backward compatibility
                def __call__(self, query):
                    return self.invoke(query)

            return SimpleQAChain(vectorstore.as_retriever())


# New function to rank papers by abstract relevance
def rank_papers_by_relevance(query, papers, embedding_model):
    """Rank papers by the relevance of their abstracts to the query."""
    print("Ranking papers by abstract relevance...")

    try:
        # Get abstracts and titles
        abstracts = [paper["summary"] for paper in papers]
        titles = [paper["title"] for paper in papers]

        # Combine for context
        texts = [
            f"Title: {title}\nAbstract: {abstract}" for title, abstract in zip(titles, abstracts)
        ]

        # Embed the query
        if hasattr(embedding_model, "embed_query"):
            # LangChain embeddings
            query_embedding = embedding_model.embed_query(query)
        else:
            # SentenceTransformer model
            query_embedding = embedding_model.encode(query)

        # Embed the abstracts
        if hasattr(embedding_model, "embed_documents"):
            # LangChain embeddings
            abstract_embeddings = embedding_model.embed_documents(texts)
        else:
            # SentenceTransformer model
            abstract_embeddings = embedding_model.encode(texts)

        # Calculate similarity scores
        query_embedding = np.array(query_embedding).reshape(1, -1)
        abstract_embeddings = np.array(abstract_embeddings)

        similarities = cosine_similarity(query_embedding, abstract_embeddings)[0]

        # Create a list of (paper, similarity) tuples
        paper_scores = list(zip(papers, similarities))

        # Sort by similarity score in descending order
        ranked_papers = [
            paper for paper, score in sorted(paper_scores, key=lambda x: x[1], reverse=True)
        ]

        # Print the ranking for debugging
        print("\nPaper Ranking by Abstract Relevance:")
        for i, (paper, score) in enumerate(sorted(paper_scores, key=lambda x: x[1], reverse=True)):
            print(f"{i+1}. {paper['title']} (Score: {score:.4f})")

        return ranked_papers

    except Exception as e:
        print(f"Error ranking papers: {e}")
        # Return original order if ranking fails
        return papers


# 6. Autonomous research assistant function
def research_assistant(query):
    print("Searching for papers on arXiv...")
    papers = search_arxiv(query, max_results=MAX_INIT_RESULTS)
    if not papers:
        print("No papers found for the query.")
        return

    print(f"Found {len(papers)} papers. Extracting text...")
    texts = extract_texts_from_results(papers)

    try:
        print("Creating vector store...")
        vectorstore, embedding_model = create_vectorstore(texts)

        # Rank papers by abstract relevance
        ranked_papers = rank_papers_by_relevance(query, papers, embedding_model)

        # Take the top MAX_FINAL_RESULTS most relevant papers
        top_papers = ranked_papers[:MAX_FINAL_RESULTS]
        print(f"Selected top {len(top_papers)} most relevant papers based on abstract content.")

        # Create a new vector store with only the top papers
        top_texts = extract_texts_from_results(top_papers)
        top_vectorstore, _ = create_vectorstore(top_texts)

        print("Setting up language model...")
        llm = create_llm()

        print("Creating retrieval chain...")
        qa_chain = create_retrieval_chain(top_vectorstore, llm)

        print("Generating answer...")
        # Use invoke instead of run (which is deprecated)
        try:
            # First try the new invoke method
            response = qa_chain.invoke(query)
            result = response.get("result", "No result found")
        except AttributeError:
            # If invoke doesn't exist, try __call__
            try:
                response = qa_chain(query)
                result = response.get("result", "No result found")
            except Exception as call_error:
                print(f"Error calling chain: {call_error}")
                result = "Could not generate a response, but here are some relevant papers."

        print("\nQuery:", query)
        print("\nAnswer:", result)
        print("\nTop Relevant Papers:")
        for i, paper in enumerate(top_papers):
            print(f"{i+1}. {paper['title']} ({paper['url']})")
            print(f"   Authors: {paper['authors']}")
            print(f"   Published: {paper['published']}")
            print()
    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nEven though the QA system failed, here are the papers we found:")
        for paper in papers:
            print(f"- {paper['title']} ({paper['url']})")


def main():
    """Run the ArXiv Research Assistant as a command-line tool."""
    print("ArXiv Research Assistant")
    print("------------------------")
    print("This tool searches arXiv for papers on your topic and provides summaries.")
    print("The papers are ranked by the relevance of their abstracts to your query.")
    print("------------------------")

    while True:
        user_query = input("\nEnter your research query (or 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        research_assistant(user_query)


if __name__ == "__main__":
    main()
