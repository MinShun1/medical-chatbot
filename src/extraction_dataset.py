import json
import time

results = []

for _, row in ocr_df.iterrows():

    extracted = extract_json(row["ocr_text"])

    results.append({
        "filename": row["filename"],
        "document_type": row["document_type"],
        "prediction": extracted
    })

    time.sleep(5)

with open("data/extraction_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("Done!")
