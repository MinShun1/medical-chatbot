import os
import pandas as pd
from tqdm import tqdm

from src.ocr import run_ocr

def ocr_dataset(gt_df, image_folder):

    rows = []

    for _, row in tqdm(gt_df.iterrows(), total=len(gt_df)):

        filename = row["filename"]

        image_path = os.path.join(image_folder, filename)

        text = run_ocr(image_path)

        rows.append({
            "filename": filename,
            "document_type": row["document_type"],
            "ocr_text": text
        })

    return pd.DataFrame(rows)

bills_gt = pd.read_csv("data/medical_bills_ground_truth.csv")

bills_result = ocr_dataset(
    bills_gt,
    "data/bills"
)

bills_result.to_csv("data/ocr_results.csv", index=False)