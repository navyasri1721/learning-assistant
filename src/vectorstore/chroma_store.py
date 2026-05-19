from langchain_community.vectorstores import Chroma

from src.config import (
    PERSIST_DIRECTORY,
    CHROMA_K
)


def create_chroma_vectorstore(
    split_docs,
    embeddings
):

    # Load existing DB
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    # Add only new documents
    vectorstore.add_documents(
        split_docs
    )

    return vectorstore


def load_chroma_vectorstore(
    embeddings
):

    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    return vectorstore


def get_chroma_retriever(
    vectorstore
):

    return vectorstore.as_retriever(

        search_type="similarity_score_threshold",

        search_kwargs={
            "k": CHROMA_K,
            "score_threshold": 0.5
        }
    )