from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chunking.chunking import ChunkingStrategy
from langchain.docstore.document import Document

class RecursiveCharacterTextSplitterStrategy(ChunkingStrategy):
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        return self.text_splitter.split_documents(documents)