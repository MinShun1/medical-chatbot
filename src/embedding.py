from google import genai

client = None

def initialize_client(api_key):
    global client
    client = genai.Client(api_key=api_key)


def embed_text(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return response.embeddings[0].values
