from langchain_groq import ChatGroq


class LLMFactory:

    @staticmethod
    def create_llm(api_key):

        print("LLM Factory Called")

        return ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.1-8b-instant"
        )