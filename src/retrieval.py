import faiss
import pickle
import numpy as np

from src.embedding import embed_text


class Retriever:

    def __init__(self, index_path, metadata_path):

        self.index = faiss.read_index(index_path)

        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

    def retrieve(self, question, k=7):
    
        query_vector = embed_text(question)
    
        query = np.array([query_vector]).astype("float32")
    
        distances, indices = self.index.search(query, k)
    
        retrieved_docs = []
    
        for idx in indices[0]:
            retrieved_docs.append(self.metadata[idx])
    
        context = self.build_context(retrieved_docs)
    
        return retrieved_docs, context

    def build_context(self, docs):

        context = ""

        for doc in docs:

            context += doc["text"]
            context += "\n\n"

        return context
