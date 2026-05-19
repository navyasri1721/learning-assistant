from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
import streamlit as st
@st.cache_resource
class CustomHFEmbeddings(Embeddings):

    def __init__(self):

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def embed_documents(self, texts):

        return self.model.encode(
            texts
        ).tolist()

    def embed_query(self, text):

        if text is None or not str(text).strip():

            return [0.0] * 384

        return self.model.encode(
            str(text)
        ).tolist()