from src.pipeline.base_handler import (
    BaseHandler
)

from src.reranker.reranker import (
    rerank_documents
)


class RerankHandler(BaseHandler):

    def handle(self, data):

        print("Rerank Handler Running")

        reranked_docs = rerank_documents(
            data["question"],
            data["docs"]
        )

        data["docs"] = reranked_docs

        return super().handle(data)