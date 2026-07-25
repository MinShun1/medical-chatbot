import json
import os
from google import genai

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def extract_json(document):

    prompt = f"""
You are an expert medical information extraction system.

Extract information from the OCR text below.

Return ONLY valid JSON.

If a value is missing, return null.

Schema:

{{
  "hospital": {{
      "name": "",
      "address": "",
      "npi": ""
  }},
  "patient": {{
      "name": "",
      "dob": "",
      "mrn": ""
  }},
  "diagnosis": "",
  "icd_10": "",
  "hospital_course": "",
  "procedures": [],
  "medications": [],
  "insurance": "",
  "financials": {{
      "total_charges": 0,
      "insurance_adjustment": 0,
      "patient_responsibility": 0
  }}
}}

Medical document:

{document}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    text = response.text

    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

    return json.loads(text)