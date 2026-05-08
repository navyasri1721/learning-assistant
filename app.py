import streamlit as st

from langchain_groq import ChatGroq

from langchain_text_splitters import CharacterTextSplitter

from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.documents import Document

st.set_page_config(
    page_title="AI Personal Learning Assistant",
    layout="centered"
)
st.title(" AI Personal Learning Assistant")

st.markdown(
    """
Ask questions related to:
- Programming
- Artificial Intelligence
- Interview Preparation
- Career Guidance
"""
)
llm = ChatGroq(
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model_name="llama-3.3-70b-versatile"
)
learning_data = """

Python is a programming language.

Artificial Intelligence enables machines to think.

Machine Learning is a subset of AI.

DBMS stands for Database Management System.

"""

documents = [Document(page_content=learning_data)]

text_splitter = CharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(
    docs,
    embeddings
)
retriever = vectorstore.as_retriever()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

user_question = st.chat_input("Ask your question here...")

if user_question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )
    with st.chat_message("user"):

        st.markdown(user_question)
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):
            retrieved_docs = retriever.invoke(user_question)
            context = "\n".join(
                [doc.page_content for doc in retrieved_docs]
            )
            prompt = f"""
You are an AI Personal Learning Assistant.

Answer the user's question clearly and simply.

Context:
{context}

Question:
{user_question}
"""
            response = llm.invoke(prompt)

            ai_response = response.content
            st.markdown(ai_response)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": ai_response
        }
    )
