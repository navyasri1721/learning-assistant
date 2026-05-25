from abc import ABC, abstractmethod


class RetrievalStrategy(ABC):

    @abstractmethod
    def retrieve(self, query):
        pass