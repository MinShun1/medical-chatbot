import re

from google import genai

from src.prompt import SYSTEM_PROMPT
from src.retrieval import Retriever


class MedicalChatbot:

    def __init__(self, api_key, index_path, metadata_path):

        self.client = genai.Client(api_key=api_key)

        self.retriever = Retriever(
            index_path=index_path,
            metadata_path=metadata_path
        )

    def ask(self, question, k=7):

        # =====================
        # Exact filename lookup
        # =====================

        match = re.search(r"(med_doc_[\w\d_]+\.jpg)", question)

        if match:

            filename = match.group(1)

            docs = [
                x
                for x in self.retriever.metadata
                if x["filename"] == filename
            ]

            context = ""

            for doc in docs:

                context += doc["text"]
                context += "\n\n"

        else:

            docs, context = self.retriever.retrieve(question, k)

        prompt = f"""
{SYSTEM_PROMPT}

Retrieved Documents:

{context}

Question:

{question}
"""

        response = self.client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )

        return response.text