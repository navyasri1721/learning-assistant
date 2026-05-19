import tempfile
import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader
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


def process_uploaded_files(uploaded_files):

    all_docs = []

    for uploaded_file in uploaded_files:

        file_extension = (
            uploaded_file.name.split(".")[-1]
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{file_extension}"
        ) as tmp_file:

            tmp_file.write(
                uploaded_file.read()
            )

            temp_path = tmp_file.name

        documents = load_document(
            temp_path,
            file_extension
        )

        for doc in documents:

            doc.metadata["source"] = (
                uploaded_file.name
            )

        all_docs.extend(documents)

        os.remove(temp_path)

    return all_docs