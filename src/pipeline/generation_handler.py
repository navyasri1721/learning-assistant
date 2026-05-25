from src.pipeline.base_handler import (
    BaseHandler
)

from src.refiner.context_refiner import (
    refine_context
)

from langchain_core.prompts import (
    PromptTemplate
)

from src.prompts.rag_prompt import (
    RAG_PROMPT
)


class GenerationHandler(BaseHandler):

    def __init__(self, llm, memory):

        super().__init__()

        self.llm = llm
        self.memory = memory

    def handle(self, data):

        print("Generation Handler Running")

        context = refine_context(
            data["docs"]
        )

        prompt = PromptTemplate(
            template=RAG_PROMPT,
            input_variables=[
                "chat_history",
                "context",
                "question"
            ]
        )

        final_prompt = prompt.format(
            chat_history=str(
                self.memory.buffer
            ),
            context=context,
            question=data["question"]
        )

        response = self.llm.invoke(
            final_prompt
        )

        data["answer"] = response.content

        return super().handle(data)