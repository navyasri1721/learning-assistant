from langchain_core.prompts import PromptTemplate


def rewrite_query(
    question,
    memory,
    llm
):

    history = memory.buffer

    prompt = PromptTemplate(

        input_variables=[
            "history",
            "question"
        ],

        template="""
You are a query rewriting assistant.

Your task:
- Convert follow-up questions into standalone questions.
- Keep the meaning EXACTLY same.
- Do NOT add extra information.
- Do NOT explain anything.
- Return ONLY the rewritten question.

Chat History:
{history}

Current Question:
{question}

Rewritten Question:
"""
    )

    final_prompt = prompt.format(

        history=history,

        question=question
    )

    response = llm.invoke(
        final_prompt
    )

    return response.content.strip()
