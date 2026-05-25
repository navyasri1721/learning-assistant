from src.strategies.retrieval_strategy import (
    RetrievalStrategy
)

class HybridStrategy(RetrievalStrategy):

    def __init__(self, retriever):

        self.retriever = retriever

    def retrieve(self, query):
        print("Hybrid Strategy Used")
        return self.retriever.invoke(query)