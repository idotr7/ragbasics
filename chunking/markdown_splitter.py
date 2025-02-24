from typing import List
from langchain_text_splitters import MarkdownHeaderTextSplitter
from chunking.chunking import ChunkingStrategy
from langchain.docstore.document import Document


class MarkdownHeaderTextSplitterStrategy(ChunkingStrategy):
    def __init__(self, headers_to_split_on: List[str]):
        self.text_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            return_each_line=True,
            strip_headers=False,
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        return self.text_splitter.split_text(documents)
