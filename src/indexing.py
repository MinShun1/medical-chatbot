import faiss
import numpy as np
import pickle

def json_to_document(item):

    pred = item["prediction"]

    hospital = pred.get("hospital", {})
    patient = pred.get("patient", {})
    financials = pred.get("financials", {})

    procedures = ", ".join(
        p if isinstance(p, str) else p.get("desc", "")
        for p in pred.get("procedures", [])
    )

    medications = ", ".join(pred.get("medications", []))

    return f"""
Filename: {item['filename']}

Hospital: {hospital.get('name')}
Address: {hospital.get('address')}
NPI: {hospital.get('npi')}

Patient: {patient.get('name')}
DOB: {patient.get('dob')}
MRN: {patient.get('mrn')}

Diagnosis: {pred.get('diagnosis')}
ICD-10: {pred.get('icd_10')}

Hospital Course:
{pred.get('hospital_course')}

Procedures:
{procedures}

Medications:
{medications}

Insurance:
{pred.get('insurance')}

Total Charges:
{financials.get('total_charges')}

Insurance Adjustment:
{financials.get('insurance_adjustment')}

Patient Responsibility:
{financials.get('patient_responsibility')}
""".strip()

def build_faiss_index(embeddings):

    vectors = np.array(embeddings).astype("float32")

    dimension = vectors.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(vectors)

    return index


def save_index(index, path):

    faiss.write_index(index, path)


def save_metadata(metadata, path):

    with open(path, "wb") as f:
        pickle.dump(metadata, f)