import json
import time

from tqdm import tqdm

from src.embedding import embed_text
from src.indexing import (
    json_to_document,
    build_faiss_index,
    save_index,
    save_metadata
)

with open("data/extraction_results.json", "r") as f:
    docs = json.load(f)

documents = []

for item in docs:

    documents.append({
        "filename": item["filename"],
        "text": json_to_document(item)
    })

embeddings = []

for i, doc in enumerate(tqdm(documents)):

    embeddings.append(embed_text(doc["text"]))

    if (i + 1) % 90 == 0:
        time.sleep(60)

index = build_faiss_index(embeddings)

save_index(index, "data/medical_index.faiss")

metadata = []

for doc in documents:

    metadata.append({
        "filename": doc["filename"],
        "text": doc["text"],
        "prediction": next(
            x["prediction"]
            for x in docs
            if x["filename"] == doc["filename"]
        )
    })

save_metadata(metadata, "data/metadata.pkl")
