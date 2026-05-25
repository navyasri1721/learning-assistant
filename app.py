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
from src.pipeline.query_rewrite_handler import (
    QueryRewriteHandler
)

from src.pipeline.retrieval_handler import (
    RetrievalHandler
)

from src.pipeline.rerank_handler import (
    RerankHandler
)

from src.pipeline.generation_handler import (
    GenerationHandler
)

# ---------------- QUERY REWRITE ----------------

from src.query_rewrite.query_rewriter import (
    rewrite_query
)

# ---------------- RERANK ----------------

from src.reranker.reranker import (
    rerank_documents
)

# ---------------- REFINE ----------------

from src.refiner.context_refiner import (
    refine_context
)

# ---------------- MYSQL ----------------

from src.database.mysql_connection import (
    save_rag_data,
    fetch_rag_data
)

# ---------------------------------------------------
# STREAMLIT PAGE
# ---------------------------------------------------

st.set_page_config(
    page_title="Hybrid RAG System",
    layout="wide"
)

st.title("Hybrid RAG System")

# ---------------------------------------------------
# FETCH MYSQL DATA
# ---------------------------------------------------

fetch_rag_data()

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []

if "retriever" not in st.session_state:

    st.session_state.retriever = None

if "memory" not in st.session_state:

    st.session_state.memory = get_memory()

if "processed_files" not in st.session_state:

    st.session_state.processed_files = []

# ---------------------------------------------------
# LLM
# ---------------------------------------------------

from src.factories.llm_factory import (
    LLMFactory
)

llm = LLMFactory.create_llm(
    st.secrets["GROQ_API_KEY"]
)

# ---------------------------------------------------
# EMBEDDINGS
# ---------------------------------------------------

from src.singleton.embedding_singleton import (
    EmbeddingSingleton
)

embeddings = (
    EmbeddingSingleton.get_instance()
)

# ---------------------------------------------------
# LOAD EXISTING DATABASE
# ---------------------------------------------------

if (
    os.path.exists(PERSIST_DIRECTORY)
    and st.session_state.retriever is None
):

    try:

        vectorstore = load_chroma_vectorstore(
            embeddings
        )

        # LOAD ALL DOCUMENTS
        all_docs = vectorstore.get()

        documents = []

        if all_docs and all_docs["documents"]:

            from langchain_core.documents import (
                Document
            )

            for i in range(
                len(all_docs["documents"])
            ):

                doc = Document(

                    page_content=all_docs[
                        "documents"
                    ][i],

                    metadata=all_docs[
                        "metadatas"
                    ][i]
                )

                documents.append(doc)

        # CHROMA RETRIEVER
        chroma_retriever = (
            get_chroma_retriever(
                vectorstore
            )
        )

        # HYBRID RETRIEVER
        retriever = create_hybrid_retriever(
            documents,
            chroma_retriever
        )

        st.session_state.retriever = (
            retriever
        )

        st.sidebar.success(
            "Persistent Database Loaded"
        )

    except Exception as e:

        st.sidebar.error(
            f"Error loading DB: {e}"
        )

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title(
    "Upload Documents"
)

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
# PROCESS DOCUMENTS
# ---------------------------------------------------

if uploaded_files:

    uploaded_file_names = sorted([

        file.name

        for file in uploaded_files
    ])

    if (
        uploaded_file_names
        != st.session_state.processed_files
    ):

        with st.spinner(
            "Processing documents..."
        ):

            # LOAD DOCUMENTS
            docs = process_uploaded_files(
                uploaded_files
            )

            # SPLIT DOCUMENTS
            split_docs = split_documents(
                docs
            )

            # SAVE TO CHROMA
            vectorstore = (
                create_chroma_vectorstore(
                    split_docs,
                    embeddings
                )
            )

            # LOAD ALL DOCS AGAIN
            all_docs = vectorstore.get()

            documents = []

            if all_docs and all_docs["documents"]:

                from langchain_core.documents import (
                    Document
                )

                for i in range(
                    len(all_docs["documents"])
                ):

                    doc = Document(

                        page_content=all_docs[
                            "documents"
                        ][i],

                        metadata=all_docs[
                            "metadatas"
                        ][i]
                    )

                    documents.append(doc)

            # CHROMA RETRIEVER
            chroma_retriever = (
                get_chroma_retriever(
                    vectorstore
                )
            )

            # HYBRID RETRIEVER
            retriever = (
                create_hybrid_retriever(
                    documents,
                    chroma_retriever
                )
            )

            st.session_state.retriever = (
                retriever
            )

            st.session_state.processed_files = (
                uploaded_file_names
            )

        st.success(
            "Documents Processed Successfully"
        )

# ---------------------------------------------------
# CLEAR CHAT
# ---------------------------------------------------

if st.sidebar.button(
    "Clear Chat"
):

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

    st.session_state.messages.append({

        "role": "user",

        "content": user_question
    })

    with st.chat_message("user"):

        st.markdown(user_question)

    with st.chat_message("assistant"):

        if st.session_state.retriever is None:

            st.warning(
                "Please upload documents first."
            )

        else:

            try:

                # ---------------------------------------------------
                # CREATE HANDLERS
                # ---------------------------------------------------

                rewrite_handler = (
                    QueryRewriteHandler(
                        llm,
                        st.session_state.memory
                    )
                )

                retrieval_handler = (
                    RetrievalHandler(
                        st.session_state.retriever
                    )
                )

                rerank_handler = (
                    RerankHandler()
                )

                generation_handler = (
                    GenerationHandler(
                        llm,
                        st.session_state.memory
                    )
                )

                # ---------------------------------------------------
                # CONNECT HANDLERS
                # ---------------------------------------------------

                rewrite_handler.set_next(
                    retrieval_handler
                ).set_next(
                    rerank_handler
                ).set_next(
                    generation_handler
                )

                # ---------------------------------------------------
                # RUN PIPELINE
                # ---------------------------------------------------

                data = {

                    "question": user_question
                }

                result = rewrite_handler.handle(
                    data
                )

                # ---------------------------------------------------
                # GET FINAL DATA
                # ---------------------------------------------------

                answer = result["answer"]

                docs = result["docs"]

            except Exception as e:

                st.error(
                    f"Pipeline Error: {e}"
                )

                st.stop()

            # ---------------------------------------------------
            # EMPTY CONTEXT
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
            # SAVE TO MYSQL
            # ---------------------------------------------------

            save_rag_data(
                user_question,
                answer
            )

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
            # SOURCES
            # ---------------------------------------------------

            if docs:

                st.markdown("### Sources")

                shown_sources = set()

                for doc in docs[:3]:

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
                        f"{source} — Page {page}"
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