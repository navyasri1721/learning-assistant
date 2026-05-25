from src.pipeline.base_handler import (
    BaseHandler
)

from src.query_rewrite.query_rewriter import (
    rewrite_query
)


class QueryRewriteHandler(BaseHandler):

    def __init__(self, llm, memory):

        super().__init__()

        self.llm = llm
        self.memory = memory

    def handle(self, data):

        print("Rewrite Handler Running")

        rewritten_question = rewrite_query(
            question=data["question"],
            memory=self.memory,
            llm=self.llm
        )

        data["question"] = rewritten_question

        return super().handle(data)