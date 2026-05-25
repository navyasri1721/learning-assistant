from src.embeddings.huggingface_embeddings import CustomHFEmbeddings


class EmbeddingSingleton:

    _instance = None

    @classmethod
    def get_instance(cls):

        if cls._instance is None:

            cls._instance = (
                CustomHFEmbeddings()
            )

        return cls._instance