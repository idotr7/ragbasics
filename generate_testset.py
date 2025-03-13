import logging
from typing import List
import os

from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.persona import Persona
from ragas.testset.transforms import (
    HeadlineSplitter,
    NERExtractor,
    SummaryExtractor,
    EmbeddingExtractor,
    ThemesExtractor
)
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from ragas.testset.transforms.extractors.llm_based import NERExtractor
from ragas.testset.transforms.splitters import HeadlineSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def load_documents(path: str) -> List[Document]:
    """Load multiple PDF files"""
    loader = DirectoryLoader(path)
    docs = loader.load()
    return docs

async def adapt_query_types_to_hebrew(llm: LangchainLLMWrapper) -> List[tuple]:
    """Adapt query types to Hebrew language"""
    synthesizer = SingleHopSpecificQuerySynthesizer(llm=llm)
    
    # Adapt prompts to Hebrew
    prompts = await synthesizer.adapt_prompts("hebrew", llm=llm)
    synthesizer.set_prompts(**prompts)
    
    # Define distribution of query types
    return [(synthesizer, 1.0)]  # Using only single-hop queries for simplicity

def main():
    # File paths - adjust these to your PDF locations
    pdfs_path = "/Users/idotadmor/Downloads/Ono data"
    
    # Load documents
    logger.info("Loading documents...")
    docs = load_documents(pdfs_path)
    
    # Initialize models
    logger.info("Initializing models...")
    generator_llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4o", temperature=0.2)
    )
    generator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )
    
    # Setup transforms for Hebrew content
    transforms = [
        HeadlineSplitter(),
        NERExtractor()
    ]
    
    # Initialize test generator with Hebrew personas
    logger.info("Initializing test generator...")
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
    )
    
    # Generate Hebrew testset
    logger.info("Generating testset...")
    dataset = generator.generate_with_langchain_docs(
        docs,
        testset_size=10,  # Adjust size as needed
        transforms=transforms
    )
    
    # Save the testset
    logger.info("Saving testset...")
    dataset.save("rag_testset.json")
    
    # Display sample queries
    df = dataset.to_pandas()
    dataset.upload()
    logger.info("\nSample generated queries:")
    for i, row in df.iterrows():
        logger.info(f"\nQuery {i+1}: {row['question']}")
        logger.info(f"Context: {row['context'][:200]}...")  # Show first 200 chars
        logger.info(f"Answer: {row['answer']}\n")
        logger.info("-" * 80)

if __name__ == "__main__":
    main()