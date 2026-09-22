# Medical Document RAG Assistant

A Retrieval-Augmented Generation (RAG) application for querying and summarizing medical records using Optical Character Recognition (OCR), semantic embeddings, vector similarity search, and a generative language model.

The system processes image-based medical records, extracts structured information, indexes the resulting document representations, and retrieves relevant records to provide context-aware responses to user queries.

> **Disclaimer:** This project is intended for educational and research purposes only. The generated responses are not intended to provide medical diagnosis, treatment, or professional medical advice.

---

## Overview

Medical records are often stored as scanned or image-based documents, making direct information retrieval difficult. This project implements an end-to-end pipeline that transforms these documents into searchable representations and enables natural-language interaction with the resulting knowledge base.

The system combines:

* **Tesseract OCR** for text extraction
* **Information extraction** using spaCy and regular expressions
* **Gemini Embedding 001** for semantic document representation
* **FAISS** for vector similarity search
* **Gemini 3.5 Flash-Lite** for response generation
* **Streamlit** for the user interface

The current dataset consists of **1,000 medical records**.

---

## System Architecture

```text
                         Medical Record Images
                                  │
                                  ▼
                            ┌───────────┐
                            │ Tesseract │
                            │    OCR    │
                            └─────┬─────┘
                                  │
                                  ▼
                         Extracted Document Text
                                  │
                                  ▼
                       ┌────────────────────┐
                       │ Information        │
                       │ Extraction         │
                       │ spaCy + Regex      │
                       └─────────┬──────────┘
                                 │
                                 ▼
                          Document Representation
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Gemini Embedding 001     │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                         Vector Embeddings
                                 │
                                 ▼
                           ┌──────────┐
                           │  FAISS   │
                           │  Index   │
                           └────┬─────┘
                                │
                         Similarity Search
                                │
                                ▼
User Query ───────────────► Retriever
                                │
                                ▼
                         Relevant Documents
                                │
                                ▼
                    ┌──────────────────────────┐
                    │ Gemini 3.5 Flash-Lite    │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                           Generated Answer
```

---

## Retrieval Pipeline

The application implements two retrieval strategies depending on the query.

### Semantic Retrieval

For general questions, the query is transformed into an embedding and compared against the FAISS index.

```text
User Query
    │
    ▼
Query Embedding
    │
    ▼
FAISS Similarity Search
    │
    ▼
Top-k Relevant Documents
    │
    ▼
Context Construction
    │
    ▼
Gemini 3.5 Flash-Lite
    │
    ▼
Answer
```

The default retrieval parameter is `k=7`.

### Exact Document Retrieval

Queries containing a specific medical document filename are handled through an exact metadata lookup rather than semantic retrieval.

For example:

```text
Summarize med_doc_bill_100001_noisy.jpg
```

The system detects the filename and retrieves the corresponding document directly from the stored metadata.

This provides deterministic retrieval when the user explicitly references a known document.

---

## Document Processing

### OCR

Image-based medical records are processed using Tesseract OCR to obtain machine-readable text.

```text
Medical Record Image
        ↓
     Tesseract
        ↓
    OCR Text
```

### Information Extraction

The OCR output is further processed to extract and organize relevant information from each record.

The extraction pipeline uses:

* spaCy
* Regular expressions
* Custom document processing logic

The resulting information is serialized into JSON before the indexing stage.

---

## Embedding and Indexing

Each processed document is converted into a semantic vector using:

```text
gemini-embedding-001
```

The resulting embeddings are stored in a FAISS index.

Document metadata is maintained separately and includes:

```text
filename
text
prediction
```

The indexing process can be summarized as:

```text
extraction_results.json
        │
        ▼
Document Construction
        │
        ▼
Gemini Embedding
        │
        ▼
FAISS Index
        │
        ├── medical_index.faiss
        │
        └── metadata.pkl
```

---

## Generation

Retrieved documents are combined with a predefined system prompt and the user's question to construct the generation context.

Conceptually:

```text
System Instructions
        +
Retrieved Documents
        +
User Question
        │
        ▼
Gemini 3.5 Flash-Lite
        │
        ▼
Final Response
```

The generation component is implemented using the Google GenAI Python SDK.

---

## Technology Stack

| Component                    | Technology            |
| ---------------------------- | --------------------- |
| Language                     | Python                |
| User Interface               | Streamlit             |
| OCR                          | Tesseract             |
| NLP / Information Extraction | spaCy, Regex          |
| Embedding Model              | Gemini Embedding 001  |
| Vector Search                | FAISS                 |
| Generative Model             | Gemini 3.5 Flash-Lite |
| Data Serialization           | JSON, Pickle          |

---

## Project Structure

```text
medical-chatbot/
│
├── app.py
│
├── src/
│   ├── __init__.py
│   ├── build_index.py
│   ├── chatbot.py
│   ├── embedding.py
│   ├── extraction_dataset.py
│   ├── indexing.py
│   ├── information_extraction.py
│   ├── ocr.py
│   ├── ocr_dataset.py
│   ├── prompt.py
│   └── retrieval.py
│
├── data/
│   ├── extraction_results.json
│   ├── medical_index.faiss
│   └── metadata.pkl
│
├── index/
│   ├── medical_index.faiss
│   └── metadata.pkl
│
├── requirements.txt
└── README.md
```

### Core Modules

| Module                      | Responsibility                                 |
| --------------------------- | ---------------------------------------------- |
| `app.py`                    | Streamlit application and chat interface       |
| `chatbot.py`                | Main RAG orchestration and response generation |
| `ocr.py`                    | OCR processing                                 |
| `ocr_dataset.py`            | Batch OCR processing                           |
| `information_extraction.py` | Medical information extraction                 |
| `extraction_dataset.py`     | Batch information extraction                   |
| `embedding.py`              | Gemini embedding generation                    |
| `indexing.py`               | FAISS index construction and persistence       |
| `build_index.py`            | End-to-end embedding and indexing pipeline     |
| `retrieval.py`              | Similarity-based document retrieval            |
| `prompt.py`                 | LLM system prompt and generation instructions  |

---

## Dataset

The system currently uses **1,000 medical records** represented as image-based documents.

The processing pipeline transforms the original documents into:

1. OCR text
2. Extracted information
3. Semantic embeddings
4. FAISS vectors
5. Searchable metadata

The resulting index allows the application to retrieve relevant records without passing the entire document collection to the language model.

---

## Installation

### Clone the repository

```bash
git clone https://github.com/MinShun1/medical-chatbot.git
cd medical-chatbot
```

### Create a virtual environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure the Gemini API Key

Create:

```text
.streamlit/secrets.toml
```

and add:

```toml
GEMINI_API_KEY = "your-api-key"
```

The API key should not be committed to the repository.

---

## Running the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

---

## Building the Vector Index

The FAISS index can be generated from the extracted dataset using:

```bash
python -m src.build_index
```

The process generates the document embeddings and stores the resulting FAISS index and metadata for retrieval.

---

## Example Queries

The application supports both document-specific and general medical queries.

**Document-specific**

```text
Summarize med_doc_bill_100001_noisy.jpg
```

**Patient-related**

```text
Who is Amit Singh?
```

```text
List medications for Amit Singh.
```

**General medical knowledge**

```text
What is hypertension?
```

```text
What foods are recommended for hypertension?
```

---

## Limitations

The current implementation has several limitations:

* OCR performance depends on document quality and image readability.
* Retrieval quality depends on the quality of OCR and information extraction.
* Semantic retrieval does not guarantee that the retrieved documents contain sufficient evidence to answer a query.
* Generated responses may contain inaccurate or incomplete information.
* The current conversational interface maintains chat history at the application level, while retrieval is performed independently for each user query.
* The system has not been validated for clinical use.

---

## Future Development

Potential improvements include:

* Conversational retrieval with previous-turn context
* Hybrid lexical and semantic retrieval
* Retrieval re-ranking
* Improved document chunking strategies
* Retrieval and generation evaluation metrics
* OCR preprocessing and quality enhancement
* Source attribution for retrieved documents
* Document-level evidence visualization
* Hallucination and faithfulness evaluation

---

## Disclaimer

This application is developed for **educational and research purposes**.

The system does not provide medical diagnosis, treatment decisions, or professional medical advice. Information generated by the application should not be used as a substitute for consultation with a qualified healthcare professional.
