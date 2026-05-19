from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


def split_documents(documents):

    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
    )

    return text_splitter.split_documents(
        documents
    )