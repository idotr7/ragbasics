from typing import List
from abc import ABC, abstractmethod
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Document]:
        pass



