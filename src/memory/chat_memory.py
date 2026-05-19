from langchain_classic.memory import (
    ConversationBufferWindowMemory
)

def get_memory():

    return ConversationBufferWindowMemory(
        k=3,
        return_messages=True
    )