# Enterprise RAG Auditing & Compliance Engine

A Python-based Retrieval-Augmented Generation (RAG) application for searching enterprise policy documents, retrieving supporting evidence, and identifying potentially unsupported responses.

**Live Demo:** https://enterprise-rag-auditing-engine.streamlit.app

**Source Code:** https://github.com/Chandana-B6/Enterprise-RAG-Auditing-Engine

## Overview

Enterprise organizations rely on policy documents containing important rules and procedures. Finding relevant information manually can be time-consuming.

This project demonstrates a document-grounded retrieval system that allows users to upload PDF policy documents and ask questions about their contents.

The application retrieves relevant document passages, generates extractive answers, displays supporting context, and abstains when sufficiently relevant evidence cannot be retrieved.

## Key Features

- PDF document ingestion and text extraction
- Sentence-aware text chunking
- Semantic embeddings using Sentence Transformers
- Vector similarity search with ChromaDB
- Extractive, document-grounded answers
- Retrieved source-context display
- Relevance-based abstention for unsupported queries
- Basic heuristic compliance auditing
- Interactive Streamlit interface

## Technology Stack

- Python
- Streamlit
- ChromaDB
- Sentence Transformers
- PyPDF
- NumPy

## Architecture

PDF Upload → Text Extraction → Sentence-aware Chunking → Embedding Generation → ChromaDB → Semantic Retrieval → Extractive Answer → Compliance Heuristic

## Example Scenarios

**Supported question:**  
"What must employees do when they suspect a security incident?"

Expected behavior: Retrieve the incident-reporting policy and display its supporting text.

**Unsupported question:**  
"How many days of paid maternity leave does the company provide?"

Expected behavior: Abstain when the uploaded policy does not provide sufficiently relevant evidence.

## Running Locally

Clone the repository:

```bash
git clone https://github.com/Chandana-B6/Enterprise-RAG-Auditing-Engine.git
cd Enterprise-RAG-Auditing-Engine
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Limitations

- Answer generation uses extractive retrieval rather than an external LLM.
- Relevance filtering and compliance auditing use heuristic methods.
- The compliance score is not a validated probability of hallucination.
- Retrieval quality depends on document quality, chunking, and threshold configuration.
- Persistent audit logging and large-scale evaluation are areas for future development.

## Future Improvements

- Persistent audit history
- Automated retrieval evaluation
- Improved source attribution, including page-level citations
- Hybrid keyword and semantic retrieval
- More robust factual consistency verification

## Author

Developed as a practical project exploring document intelligence, semantic retrieval, and compliance-oriented AI systems.
