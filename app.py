import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
import tempfile
import os
st.set_page_config(
    page_title="AI Personal Learning Assistant",
    layout="wide"
)
st.title(" AI Personal Learning Assistant")
llm = ChatGroq(
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model_name="llama-3.3-70b-versatile"
)
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Files",
    type="pdf",
    accept_multiple_files=True
)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if uploaded_files:
    all_docs = []
    with st.spinner("Processing PDFs..."):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            for doc in documents:
                doc.metadata["source"] = uploaded_file.name
            all_docs.extend(documents)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        split_docs = text_splitter.split_documents(all_docs)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_documents(
            split_docs,
            embeddings
        )
        st.session_state.vectorstore = vectorstore
    st.success("PDFs Uploaded & Processed Successfully!")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
user_question = st.chat_input(
    "Ask questions from uploaded PDFs..."
)
if user_question:
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })
    with st.chat_message("user"):
        st.markdown(user_question)
    with st.chat_message("assistant"):
        if st.session_state.vectorstore is None:
            st.warning("Please upload PDF files first.")
        else:
            with st.spinner("Thinking..."):
                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 5}
                )

                docs = retriever.invoke(user_question)
                context = "\n\n".join(
                    [doc.page_content for doc in docs]
                )

                prompt_template = """
You are an AI Personal Learning Assistant.
Answer the user's question ONLY from the provided context.
If the answer exists in the PDFs:
- Explain clearly
- Give detailed answer
- Mention important points
- Include source PDF name and page number
If the answer is not available, say:
"I could not find this information in the uploaded PDFs."
Context:
{context}
Question:
{question}
Answer:
"""
                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )
                final_prompt = prompt.format(
                    context=context,
                    question=user_question
                )
                response = llm.invoke(final_prompt)
                answer = response.content
                st.markdown(answer)
                st.markdown("Sources")
                shown_sources = set()
                for doc in docs:
                    source = doc.metadata.get("source", "Unknown File")
                    page = doc.metadata.get("page", "Unknown Page")
                    source_text = f"📄 {source} — Page {page + 1}"
                    if source_text not in shown_sources:
                        st.markdown(f"- {source_text}")
                        shown_sources.add(source_text)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })