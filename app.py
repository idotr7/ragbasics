import asyncio
import logging
import time
from typing import Any, Iterator, Dict, Optional

import gradio as gr

from langchain_chroma import Chroma
from langchain import hub  # type: ignore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, Runnable, RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_voyageai import VoyageAIEmbeddings
from chunking import Chunker

# Import the prompt caching implementations
from LLM import (
    OpenAIWithCache,
    GeminiWithCache,
    ClaudeWithCache,
    preload_ono_data_to_cache
)

from toml import load  # type: ignore

from fastapi import FastAPI
from dotenv import load_dotenv
import os

load_dotenv()

# os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
# google_api_key = os.getenv("GOOGLE_API_KEY")
# print(google_api_key)
app = FastAPI()


@app.get("/")
def greet_json():
    return {"Hello": "World!"}


# load pyproject.toml
with open("pyproject.toml", "r") as f:
    config = load(f)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global objects
vector_store = None
cache_enabled = False
llm_provider = "google"  # Default provider
cached_llm = None

# Initialize language model and embeddings with streaming capability
# llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)  # OpenAI
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key="AIzaSyA0IOAcvniHJArb_FIeW_2ryqxNMFPu1OM") # Google

# embedding = OpenAIEmbeddings(model="text-embedding-3-small")
embedding = VoyageAIEmbeddings(
    model="voyage-3-lite", 
    output_dimension=512,
    batch_size=128  # Adding required batch_size parameter
)

# Initialize cached LLM based on provider
def init_cached_llm(provider: str = "google") -> None:
    """
    Initialize the cached LLM based on the specified provider.
    
    Args:
        provider (str): The LLM provider to use ('openai', 'google', or 'anthropic')
    """
    global cached_llm, llm_provider
    
    llm_provider = provider.lower()
    
    if llm_provider == "openai":
        logger.info("Initializing OpenAI with cache...")
        cached_llm = OpenAIWithCache(model="gpt-4o-mini")
    elif llm_provider == "google":
        logger.info("Initializing Google Gemini with cache...")
        cached_llm = GeminiWithCache(model="gemini-2.0-flash")
    elif llm_provider == "anthropic":
        logger.info("Initializing Anthropic Claude with cache...")
        cached_llm = ClaudeWithCache(model="claude-3-sonnet-20240229")
    else:
        logger.error(f"Unknown LLM provider: {llm_provider}")
        cached_llm = None

# Optimize vector store configuration (to be called after creating vector store)
def optimize_vector_store():
    """Apply optimizations to the vector store configuration"""
    if vector_store is None:
        return

    # For Chroma, you might set search parameters like this
    # These are example parameters - actual optimal values depend on your data
    # vector_store._collection_metadata["hnsw:space"] = "cosine"
    # vector_store._collection_metadata["hnsw:M"] = 16
    # vector_store._collection_metadata["hnsw:ef_construction"] = 100
    # vector_store._collection.metadata = vector_store._collection_metadata
    logger.info("Vector store optimized for faster retrieval")


def load_files(file_paths: list[str], enable_cache: bool = False, cache_provider: str = "google") -> str:
    """
    Load multiple PDF files, split them into chunks,
    and store these chunks in a global vector store (Chroma).
    Optionally enables prompt caching.

    Args:
        file_paths (list[str]): List of paths to PDF files.
        enable_cache (bool): Whether to enable prompt caching.
        cache_provider (str): Which LLM provider to use for caching.

    Returns:
        str: A status message indicating the files were loaded successfully.
    """
    global vector_store, cache_enabled
    all_chunks = []

    for file_path in file_paths:
        logger.info(f"Loading file: {file_path}")

        # Load the document
        document_loader = PyMuPDFLoader(file_path)
        documents = document_loader.load()
        logger.info(f"Loaded {len(documents)} document(s) from {file_path}")

        # Split the document into chunks
        chunks = Chunker().chunk(documents)
        logger.info(f"Split into {len(chunks)} chunk(s)")

        all_chunks.extend(chunks)

        logger.info(f"Total chunks across all documents: {len(all_chunks)}")

    logger.info("Creating vector store...")
    vector_store = Chroma.from_documents(documents=all_chunks, embedding=embedding)

    logger.info("Vector store created successfully.")
    
    # Initialize prompt caching if requested
    if enable_cache:
        cache_enabled = True
        init_cached_llm(cache_provider)
        
        # Preload cache with up to 10,000 tokens from the documents
        logger.info(f"Preloading cache with LLM provider: {cache_provider}")
        preload_ono_data_to_cache(cache_provider, max_tokens=10000)
        
        return f"Loaded {len(file_paths)} files successfully. Prompt caching enabled with {cache_provider}. You can now ask questions about the documents."
    else:
        cache_enabled = False
        return f"Loaded {len(file_paths)} files successfully. You can now ask questions about the documents."


def respond(message: str, history: list) -> Iterator[str]:
    """
    Generate a streaming response based on the user query (message),
    retrieving relevant context from the global vector store.
    Uses prompt caching if enabled.

    Args:
        message (str): The user's query or message.
        history (list): Chat history (unused in this code, but required by Gradio).

    Yields:
        str: Chunks of the generated response for streaming.
    """
    # Start total response time measurement
    start_time = time.time()
    ttfb_recorded = False
    ttfb = 0.0  # Initialize as float to avoid type error

    # Check if vector store is initialized
    if vector_store is None:
        logger.warning("Vector store is not initialized. Please upload a PDF first.")
        yield "No document loaded. Please upload a PDF first."
        return
    
    # If caching is enabled and we have a cached response, use it
    if cache_enabled and cached_llm is not None:
        try:
            # Get the cached response (or generate a new one and cache it)
            response = cached_llm.get_completion(message)
            yield response
            logger.info(f"Total response time with caching: {time.time() - start_time:.2f}s")
            return
        except Exception as e:
            logger.error(f"Error with cached LLM: {e}")
            logger.info("Falling back to standard RAG approach...")
            # Continue with standard RAG approach

    # Standard RAG approach if caching is not enabled or failed
    # Step 1: Get the prompt template
    prompt = hub.pull("rlm/rag-prompt")

    # Step 2: Run retrieval
    try:
        # Embed the query
        embed_start_time = time.time()
        embed_query = embedding.embed_query(message)
        embed_time = time.time() - embed_start_time
        logger.info(f"Embedding time: {embed_time:.2f} seconds")

        # Retrieve relevant documents
        retrieval_start_time = time.time()
        retrieved_docs = vector_store.similarity_search_by_vector(embed_query)
        retrieval_time = time.time() - retrieval_start_time
        logger.info(
            f"Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f} seconds"
        )
        
        # Step 3: Prepare the prompt with retrieved context
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
        formatted_prompt = prompt.invoke({"question": message, "context": docs_content})

        # Step 4: Stream the LLM response
        prompt_ready_time = time.time()

        # Get the streaming response
        streaming_response = llm.stream(formatted_prompt)

        # Track accumulated response for Gradio's streaming interface
        accumulated_response = ""
        first_chunk = True

        for chunk in streaming_response:
            # Extract content from chunk, ensuring it's a string
            chunk_content = chunk.content if hasattr(chunk, "content") else str(chunk)

            # Record time to first byte (TTFB)
            if first_chunk and not ttfb_recorded:
                ttfb = time.time() - prompt_ready_time
                ttfb_recorded = True
                first_chunk = False
                logger.info(f"Time to first byte: {ttfb:.2f}s")
            # Accumulate response - this is crucial for Gradio streaming
            # Handle both string and list/dict chunk content by converting to string
            if isinstance(chunk_content, (list, dict)):
                chunk_content = str(chunk_content)
            accumulated_response += chunk_content

            # Yield the ENTIRE accumulated response so far, not just the new chunk
            yield accumulated_response

        # Calculate and log timing information
        llm_time = time.time() - prompt_ready_time
        total_time = time.time() - start_time

        # Log detailed timing information
        logger.info(
            f"""
        Latency Breakdown:
        - Embedding: {embed_time:.2f}s
        - Document Retrieval: {retrieval_time:.2f}s
        - Time to First Byte (TTFB): {ttfb:.2f}s
        - LLM Execution: {llm_time:.2f}s
        - Total Response Time: {total_time:.2f}s
        """
        )

    except Exception as e:
        logger.error(f"Error in respond: {str(e)}")
        yield f"An error occurred: {str(e)}"


def clear_state():
    """
    Clear the global vector store reference and reset the UI fields.

    Returns:
        list: A list of values for resetting Gradio components.
    """
    global vector_store, cache_enabled, cached_llm
    vector_store = None
    cache_enabled = False
    cached_llm = None
    return [None, None, False, "google"]  # Reset file input, status textbox, cache checkbox, and provider dropdown


# Gradio UI Setup
with gr.Blocks(
    theme=gr.themes.Default(
        primary_hue="blue",
        secondary_hue="gray",
    ),
) as demo:
    gr.Markdown("# RAG with Prompt Caching")
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.Files(
                file_count="multiple", type="filepath", label="Upload PDF Documents"
            )
            
            with gr.Row():
                enable_cache_checkbox = gr.Checkbox(label="Enable Prompt Caching", value=False)
                cache_provider_dropdown = gr.Dropdown(
                    choices=["openai", "google", "anthropic"],
                    value="google",
                    label="LLM Provider for Caching"
                )
            
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                clear_btn = gr.Button("Clear")

            status_output = gr.Textbox(label="Status")

        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=respond,
                chatbot=gr.Chatbot(height=600),
                theme="soft",
                show_progress="full",
                textbox=gr.Textbox(
                    placeholder="Ask a question about the loaded documents...",
                    container=False,
                    scale=7,
                ),
            )

    def process_file_submission(files, enable_cache, cache_provider):
        """Process file submission and return status message"""
        if not files:
            return "No files selected. Please upload PDF files."
        result = load_files([f.name for f in files], enable_cache, cache_provider)
        return result

    # Set up event handlers
    submit_btn.click(
        fn=process_file_submission,
        inputs=[file_input, enable_cache_checkbox, cache_provider_dropdown],
        outputs=[status_output],
    )
    clear_btn.click(
        fn=clear_state,
        inputs=[],
        outputs=[file_input, status_output, enable_cache_checkbox, cache_provider_dropdown],
    )

# Only use this if running the script directly (not through Gradio's launch)
if __name__ == "__main__":
    demo.launch()
