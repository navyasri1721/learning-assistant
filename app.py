import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from src.config import (
    MODEL_NAME,
    PERSIST_DIRECTORY
)
from src.embeddings.huggingface_embeddings import (
    CustomHFEmbeddings
)
from src.loaders.document_loader import (
    process_uploaded_files
)
from src.vectorstore.chroma_store import (
    create_chroma_vectorstore,
    load_chroma_vectorstore,
    get_chroma_retriever
)
from src.retrievers.hybrid_retriever import (
    create_hybrid_retriever
)
from src.memory.chat_memory import (
    get_memory
)
from src.prompts.rag_prompt import (
    RAG_PROMPT
)
from src.utils.helpers import (
    split_documents
)
st.set_page_config(
    page_title="Hybrid RAG System",
    layout="wide"
)
st.title("Hybrid RAG system")

if "messages" not in st.session_state:

    st.session_state.messages = []

if "retriever" not in st.session_state:

    st.session_state.retriever = None

if "memory" not in st.session_state:

    st.session_state.memory = get_memory()

if "processed_files" not in st.session_state:

    st.session_state.processed_files = []

llm = ChatGroq(
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model_name=MODEL_NAME
)

embeddings = CustomHFEmbeddings()
if (
    os.path.exists(PERSIST_DIRECTORY)
    and st.session_state.retriever is None
):

    try:
        vectorstore = load_chroma_vectorstore(
            embeddings
        )
        retriever = get_chroma_retriever(
            vectorstore
        )
        st.session_state.retriever = retriever
        st.sidebar.success(
            "Persistent ChromaDB Loaded"
        )
    except Exception as e:
        st.sidebar.error(
            f"Error loading ChromaDB: {e}"
        )
st.sidebar.title("Upload documents")

st.sidebar.write(
    "Supported Formats:"
)
st.sidebar.write(
    "PDF | TXT | CSV | DOCX"
)
uploaded_files = st.sidebar.file_uploader(
    "Upload Files",
    type=["pdf", "txt", "csv", "docx"],
    accept_multiple_files=True
)

# ---------------------------------------------------
# PROCESS DOCUMENTS ONLY ONCE
# ---------------------------------------------------

if uploaded_files:

    uploaded_file_names = sorted([
        file.name
        for file in uploaded_files
    ])

    # Process only new uploads
    if (
        uploaded_file_names
        != st.session_state.processed_files
    ):

        with st.spinner(
            "Processing documents..."
        ):

            # Load docs
            docs = process_uploaded_files(
                uploaded_files
            )

            # Split docs
            split_docs = split_documents(
                docs
            )

            # Create / update vector DB
            vectorstore = create_chroma_vectorstore(
                split_docs,
                embeddings
            )

            # Chroma retriever
            chroma_retriever = (
                get_chroma_retriever(
                    vectorstore
                )
            )

            # Hybrid retriever
            retriever = create_hybrid_retriever(
                split_docs,
                chroma_retriever
            )

            # Save retriever
            st.session_state.retriever = (
                retriever
            )

            # Save processed files
            st.session_state.processed_files = (
                uploaded_file_names
            )

        st.success(
            "Documents Processed Successfully"
        )

# ---------------------------------------------------
# CLEAR CHAT
# ---------------------------------------------------

if st.sidebar.button("Clear Chat"):

    st.session_state.messages = []

    st.session_state.memory.clear()

    st.rerun()

# ---------------------------------------------------
# DISPLAY CHAT HISTORY
# ---------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

# ---------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------

user_question = st.chat_input(
    "Ask questions from your uploaded documents..."
)

# ---------------------------------------------------
# QUESTION PROCESSING
# ---------------------------------------------------

if user_question:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    # Show user message
    with st.chat_message("user"):

        st.markdown(user_question)

    # Assistant response
    with st.chat_message("assistant"):

        # No retriever
        if st.session_state.retriever is None:

            st.warning(
                "Please upload documents first."
            )

        else:

            try:

                docs = (
                    st.session_state.retriever.invoke(
                        user_question
                    )
                )

            except Exception as e:

                st.error(
                    f"Retriever Error: {e}"
                )

                st.stop()

            # ---------------------------------------------------
            # NO DOCUMENTS FOUND
            # ---------------------------------------------------

            if not docs:

                answer = (
                    "I could not find this "
                    "information in the "
                    "uploaded documents."
                )

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

                st.stop()

            # ---------------------------------------------------
            # BUILD CONTEXT
            # ---------------------------------------------------

            context = "\n\n".join([

                doc.page_content[:800]

                for doc in docs

                if doc.page_content
            ])

            # ---------------------------------------------------
            # LOW QUALITY CONTEXT
            # ---------------------------------------------------

            if len(context.strip()) < 50:

                answer = (
                    "I could not find this "
                    "information in the "
                    "uploaded documents."
                )

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

                st.stop()

            # ---------------------------------------------------
            # PROMPT
            # ---------------------------------------------------

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
                    st.session_state.memory.buffer
                ),
                context=context,
                question=user_question
            )

            # ---------------------------------------------------
            # LLM RESPONSE
            # ---------------------------------------------------

            response = llm.invoke(
                final_prompt
            )

            answer = response.content

            # ---------------------------------------------------
            # SAVE MEMORY
            # ---------------------------------------------------

            st.session_state.memory.save_context(
                {"input": user_question},
                {"output": answer}
            )

            # ---------------------------------------------------
            # SHOW ANSWER
            # ---------------------------------------------------

            st.markdown(answer)

            # ---------------------------------------------------
            # SAVE CHAT
            # ---------------------------------------------------

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            # ---------------------------------------------------
            # FALLBACK MESSAGE
            # ---------------------------------------------------

            fallback_message = (
                "I could not find this "
                "information in the "
                "uploaded documents."
            )

            # ---------------------------------------------------
            # SHOW SOURCES ONLY FOR VALID ANSWERS
            # ---------------------------------------------------

            if (
                answer.strip()
                != fallback_message
            ):

                if (
                    docs
                    and len(context.strip()) > 50
                ):

                    st.markdown(
                        "### Sources"
                    )

                    shown_sources = set()

                    # Show only top relevant docs
                    top_docs = docs[:2]

                    for doc in top_docs:

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

                        source_text = (
                            f"{source} — "
                            f"Page {page}"
                        )

                        if (
                            source_text
                            not in shown_sources
                        ):

                            st.markdown(
                                f"- {source_text}"
                            )

                            shown_sources.add(
                                source_text
                            )