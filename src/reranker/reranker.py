from sentence_transformers import (
    CrossEncoder
)

# Fast reranker model
reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_documents(
    question,
    docs,
    top_k=3
):

    pairs = []

    for doc in docs:

        pairs.append([
            question,
            doc.page_content
        ])

    scores = reranker_model.predict(
        pairs
    )

    scored_docs = list(
        zip(docs, scores)
    )

    scored_docs = sorted(
        scored_docs,
        key=lambda x: x[1],
        reverse=True
    )

    reranked_docs = [
        doc
        for doc, score in scored_docs[:top_k]
    ]

    return reranked_docs