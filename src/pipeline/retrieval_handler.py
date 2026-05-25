from src.pipeline.base_handler import (
    BaseHandler
)

from src.strategies.hybrid_strategy import (
    HybridStrategy
)


class RetrievalHandler(BaseHandler):

    def __init__(self, retriever):

        super().__init__()

        self.strategy = HybridStrategy(
            retriever
        )

    def handle(self, data):

        print("Retrieval Handler Running")

        docs = self.strategy.retrieve(
            data["question"]
        )

        data["docs"] = docs

        return super().handle(data)