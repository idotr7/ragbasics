import logging
from typing import List
import os

from ragas import evaluate # type: ignore
from ragas import EvaluationDataset # type: ignore
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness # type: ignore
from ragas.llms import LangchainLLMWrapper # type: ignore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from toml import load # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Load config
with open("pyproject.toml", "r") as f:
    config = load(f)

def setup_qa_chain(pdf_path: str):
    """Set up the QA chain with vector store and retriever"""
    # Load and process document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Create vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(documents=documents, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    # Set up LLM
    llm = ChatOpenAI(model="gpt-4")
    
    # Create QA chain
    prompt = ChatPromptTemplate.from_template(config["rag_prompt"]["prompt_template"])
    qa_chain = (
        {"context": lambda x: "\n".join(doc.page_content for doc in retriever.get_relevant_documents(x)), 
         "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return qa_chain, retriever

def create_evaluation_dataset(
    qa_chain,
    retriever,
    sample_queries: List[str],
    expected_responses: List[str]
) -> EvaluationDataset:
    """Create a Ragas evaluation dataset from sample queries"""
    dataset = []
    
    for query, reference in zip(sample_queries, expected_responses):
        relevant_docs = retriever.get_relevant_documents(query)
        response = qa_chain.invoke(query)
        
        dataset.append({
            "user_input": query,
            "retrieved_contexts": [doc.page_content for doc in relevant_docs],
            "response": response,
            "reference": reference,
        })
    
    return EvaluationDataset.from_list(dataset)

def main():
    # Load the generated testset
    logger.info("Loading testset...")
    evaluation_dataset = EvaluationDataset.load("rag_testset.json")
    
    # Set up evaluator LLM
    logger.info("Setting up evaluation metrics...")
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4"))
    
    # Run evaluation
    logger.info("Running evaluation...")
    results = evaluate(
        dataset=evaluation_dataset,
        metrics=[
            LLMContextRecall(),
            Faithfulness(),
            FactualCorrectness()
        ],
        llm=evaluator_llm,
    )
    
    # Log results
    logger.info("Evaluation Results:")
    logger.info(f"Context Recall: {results['context_recall']:.4f}")
    logger.info(f"Faithfulness: {results['faithfulness']:.4f}")
    logger.info(f"Factual Correctness: {results['factual_correctness']:.4f}")

if __name__ == "__main__":
    main() 