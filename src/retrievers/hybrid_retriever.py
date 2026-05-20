from langchain_community.retrievers import (
    BM25Retriever
)
from langchain_classic.retrievers import (
    EnsembleRetriever
)
from src.config import BM25_K
def create_hybrid_retriever(
    split_docs,
    chroma_retriever
):

    bm25_retriever = (
        BM25Retriever.from_documents(
            split_docs
        )
    )

    bm25_retriever.k = BM25_K

    ensemble_retriever = (
        EnsembleRetriever(
            retrievers=[
                chroma_retriever,
                bm25_retriever
            ],
            weights=[0.9, 0.1]
        )
    )
    return ensemble_retriever