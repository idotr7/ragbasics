import logging
import time

import gradio as gr

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from chunking import Chunker

from toml import load # type: ignore

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

# Global objects (kept for functionality)
vector_store = None

# Initialize language model and embeddings
llm = ChatOpenAI(model="gpt-4o-mini")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

embedding = OpenAIEmbeddings()


def load_files(file_paths: list[str]) -> str:
    """
    Load multiple PDF files, split them into chunks,
    and store these chunks in a global vector store (Chroma).

    Args:
        file_paths (list[str]): List of paths to PDF files.

    Returns:
        str: A status message indicating the files were loaded successfully.
    """
    global vector_store
    all_chunks = []

    for file_path in file_paths:
        logger.info(f"Loading file: {file_path}")

        # Load the document
        document_loader = PyPDFLoader(file_path)
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

    return f"Loaded {len(file_paths)} files successfully. You can now ask questions about the documents."


def respond(message: str, history: list) -> str:
    """
    Generate a response based on the user query (message),
    retrieving relevant context from the global vector store.

    Args:
        message (str): The user's query or message.
        history (list): Chat history (unused in this code, but required by Gradio).

    Returns:
        str: The generated response.
    """
    if vector_store is None:
        logger.warning("Vector store is not initialized. Please upload a PDF first.")
        return "No document loaded. Please upload a PDF first."

    # Start total response time measurement
    total_start_time = time.time()

    # Measure retrieval setup time
    logger.info("Setting up retriever...")
    retriever = vector_store.as_retriever()

    # Measure actual retrieval time
    logger.info("Retrieving relevant documents...")
    retrieval_start_time = time.time()
    # Force retrieval by running a similarity search
    relevant_docs = retriever.get_relevant_documents(message)
    retrieval_time = time.time() - retrieval_start_time
    logger.info(f"Retrieved {len(relevant_docs)} documents in {retrieval_time:.2f} seconds")

    # Measure chain preparation and execution time
    logger.info("Preparing and executing chain...")
    chain_start_time = time.time()
    
    prompt_template = ChatPromptTemplate.from_template(config["rag_prompt"]["prompt_template"])
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(message)
    chain_time = time.time() - chain_start_time
    
    # Calculate total time
    total_time = time.time() - total_start_time
    
    # Log detailed timing information
    logger.info(f"""
    Latency Breakdown:
    - Document Retrieval: {retrieval_time:.2f}s
    - Chain Execution: {chain_time:.2f}s
    - Total Response Time: {total_time:.2f}s
    """)

    return response


def clear_state():
    """
    Clear the global vector store reference and reset the UI fields.

    Returns:
        list: A list of values for resetting Gradio components.
    """
    global vector_store
    vector_store = None
    return [None, None]  # Reset file input and status textbox


# Gradio UI Setup
with gr.Blocks(
    theme=gr.themes.Default(
        primary_hue="blue",
        secondary_hue="gray",
    ),
) as demo:
    gr.Markdown("# RAG Starter App")
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.Files(
                file_count="multiple",
                type="filepath",
                label="Upload PDF Documents"
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
                    placeholder="Ask questions about the uploaded document!",
                    container=False,
                ),
            )

    # Set up Gradio interactions
    submit_btn.click(fn=load_files, inputs=file_input, outputs=status_output)
    clear_btn.click(
        fn=clear_state,
        outputs=[file_input, status_output],
    )

if __name__ == "__main__":
    demo.launch()
