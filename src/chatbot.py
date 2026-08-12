import re
from google import genai

from src.prompt import SYSTEM_PROMPT
from src.retrieval import Retriever
from src.embedding import initialize_client

class MedicalChatbot:

    def __init__(self, api_key, index_path, metadata_path):

        initialize_client(api_key)

        self.client = genai.Client(api_key=api_key)

        self.retriever = Retriever(
            index_path=index_path,
            metadata_path=metadata_path
        )

   def ask(self, question, k=7):

    match = re.search(r"(med_doc_[\w\d_]+\.jpg)", question)

    if match:

        filename = match.group(1)

        docs = [
            x for x in self.retriever.metadata
            if x["filename"] == filename
        ]

        context = ""

        for doc in docs:
            context += doc["text"] + "\n\n"

    else:

        docs, context = self.retriever.retrieve(question, k)

    prompt = f"""
{SYSTEM_PROMPT}

Retrieved Documents:

{context}

Question:

{question}
"""

    try:
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        print("GEMINI RESPONSE:", response)

        return response.text

    except Exception as e:
        print("GEMINI ERROR:", repr(e))
        raise e
