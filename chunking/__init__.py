from toml import load  # type: ignore
from typing import List
from chunking.recursive_character_text_splitter import RecursiveCharacterTextSplitterStrategy
from chunking.markdown_splitter import MarkdownHeaderTextSplitterStrategy
from langchain.docstore.document import Document

# load pypromect.toml
with open("pyproject.toml", "r") as f:
    config = load(f)


class Chunker:
    def __init__(self):
        if config["chunking"]["strategy"] == "recursive_character_text_splitter":
            self.chunking_strategy = RecursiveCharacterTextSplitterStrategy(
                chunk_size=config["chunking"]["chunk_size"],
                chunk_overlap=config["chunking"]["chunk_overlap"],
            )

        elif config["chunking"]["strategy"] == "markdown_splitter":
            self.chunking_strategy = MarkdownHeaderTextSplitterStrategy(
                headers_to_split_on=config["chunking"]["headers_to_split_on"],
                return_each_line=config["chunking"]["return_each_line"],
                strip_headers=config["chunking"]["strip_headers"],
            )
        else:
            raise ValueError(f"Invalid chunking strategy: {config['chunking']['strategy']}")

    def chunk(self, documents: List[Document]) -> List[Document]:
        return self.chunking_strategy.chunk(documents)
