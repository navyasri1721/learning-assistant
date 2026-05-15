import streamlit as st
import tempfile
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import (PyPDFLoader,TextLoader,CSVLoader,Docx2txtLoader)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.retrievers import EnsembleRetriever
st.set_page_config(page_title="Hybrid RAG System",layout="wide")
st.title("Hybrid RAG")
st.sidebar.title("Upload document Files")
st.sidebar.write("Supported Formats:")
st.sidebar.write("PDF | TXT | CSV | DOCX")
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
llm = ChatGroq(
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model_name="llama-3.3-70b-versatile"
)
uploaded_files = st.sidebar.file_uploader(
    "Upload Documents",
    type=["pdf", "txt", "csv", "docx"],
    accept_multiple_files=True
)
def load_document(file_path, file_type):
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == "txt":
        loader = TextLoader(file_path)
    elif file_type == "csv":
        loader = CSVLoader(file_path)
    elif file_type == "docx":
        loader = Docx2txtLoader(file_path)
    else:
        return []
    return loader.load()
@st.cache_resource
def process_documents(uploaded_files):
    all_docs = []
    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{file_extension}"
        ) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        documents = load_document(
            temp_path,
            file_extension
        )
        for doc in documents:
            doc.metadata["source"] = uploaded_file.name
        all_docs.extend(documents)
        os.remove(temp_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    split_docs = text_splitter.split_documents(all_docs)
    class CustomHFEmbeddings(Embeddings):
        def __init__(self):
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        def embed_documents(self, texts):
            return self.model.encode(texts).tolist()
        def embed_query(self, text):
            return self.model.encode(text).tolist()
    embeddings = CustomHFEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    faiss_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )
    bm25_retriever = BM25Retriever.from_documents(split_docs
    )
    bm25_retriever.k = 4
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever,bm25_retriever
        ],
        weights=[0.7, 0.3]
    )
    return ensemble_retriever
if uploaded_files:
    with st.spinner("Processing Documents..."):
        retriever = process_documents(uploaded_files)
        st.session_state.retriever = retriever
    st.success("Documents Processed Successfully")
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.rerun()
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
user_question = st.chat_input(
    "Ask questions from your uploaded documents..."
)
if user_question:

    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):

        # Greeting messages
        general_messages = [
            "hi",
            "hello",
            "hey",
            "how are you",
            "good morning",
            "good evening",
            "thank you"
        ]

        # GENERAL CHAT MODE
        if user_question.lower().strip() in general_messages:

            response = llm.invoke(user_question)

            answer = response.content

            st.write(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        # DOCUMENT QUESTION ANSWERING
        else:

            # Check if documents uploaded
            if st.session_state.retriever is None:

                st.warning("Please upload documents first.")

            else:

                with st.spinner("Thinking..."):

                    docs = st.session_state.retriever.invoke(
                        user_question
                    )

                # Build context
                context = "\n\n".join([
                    doc.page_content
                    for doc in docs
                ])

                # Prompt
                prompt_template = """
You are a helpful AI assistant with two modes:

1. GENERAL CHAT MODE:
- Respond naturally for greetings and casual chat.

2. DOCUMENT QUESTION-ANSWERING MODE:
- Answer primarily using uploaded document context.
CONVERSATION MEMORY RULES: 
- Use previous chat history to understand follow-up questions. 
- If the user asks something like: "and its types" "explain more" 
"give examples" then understand the previous topic automatically. 
- Maintain conversational continuity naturally.

RULES:
IMPORTANT RULES: - Prioritize uploaded document information 
- Give detailed and beginner-friendly explanations 
- Explain concepts clearly and step-by-step 
- If information exists in documents, answer ONLY from documents
- Do NOT invent document content 
- If information is partially available, mention that clearly IF INFORMATION IS NOT PRESENT IN DOCUMENTS:
- First say: "I could not find this information in the uploaded documents."
- Then provide a general AI-based explanation separately
- Clearly distinguish between document-based answers and general knowledge

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Answer:
"""

                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=[
                        "chat_history",
                        "context",
                        "question"
                    ]
                )

                final_prompt = prompt.format(
                    chat_history=str(
                        st.session_state.memory.buffer
                    ),
                    context=context,
                    question=user_question
                )

                # LLM response
                response = llm.invoke(final_prompt)

                answer = response.content

                # Save memory
                st.session_state.memory.save_context(
                    {"input": user_question},
                    {"output": answer}
                )

                # Show answer
                st.markdown("### Answer")
                st.write(answer)

                # Show sources ONLY if answer grounded
                if "I could not find this information" not in answer:

                    st.markdown("### Sources")

                    shown_sources = set()

                    for doc in docs:

                        source = doc.metadata.get(
                            "source",
                            "Unknown File"
                        )

                        page = doc.metadata.get(
                            "page",
                            "N/A"
                        )

                        if isinstance(page, int):
                            page = page + 1

                        source_text = f"{source} — Page {page}"

                        if source_text not in shown_sources:

                            st.markdown(f"- {source_text}")

                            shown_sources.add(source_text)

                # Save assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
