# 🛡️ Enterprise RAG Auditing & Compliance Engine

A production-ready, end-to-end Responsible AI evaluation platform engineered to safeguard retrieval-augmented generation (RAG) systems from hallucination risks, contextual token alignment failures, and non-inclusive or biased vocabulary outputs.

## 🏗️ Technical Architecture & Pipeline

```text
  [ In-App PDF Download ] ──> [ Binary PDF Extractor ] ──> [ Rolling Context Chunker ]
                                                                     │
  [ In-Memory ChromaDB ] <── [ PyTorch Vector Transducer (MiniLM) ] ─┘
            │
    (Employee Query)
            │
            ▼
  [ High-Similarity Context ] ──> [ Inference Generation Layer ]
                                                │
                                                ▼
  [ Live Compliance Audit Dashboard ] <─── [ Algorithmic Audit Guardrails ]
     ├── Hallucination Index Risk (Linguistic Overlap Scoring)
     └── Non-Inclusive / Bias Token Scanner (Ethical Governance)
```

## 🔥 System Features & Technical Achievements

*   **PyTorch-Backed Local Embeddings:** Utilizes dense mathematical transformations via `sentence-transformers/all-MiniLM-L6-v2` to maintain document data privacy inside an isolated runtime thread.
*   **Token-Level Hallucination Tracking:** Computes an algorithmic alignment ratio analyzing word intersection variance between retrieved vector context chunks and generated language output tokens.
*   **Responsible AI Core IP:** Implements proactive pattern-matching triggers to detect corporate policy governance failures, non-inclusive vocabulary, and bias markers instantly.
*   **Compliance Audit Logger:** Features a stateful persistent log ledger tracking query histories, risk variables, and violation timestamps for one-click corporate CSV export functionality.

## 🚀 Local Execution & Sandbox Setup

### 1. Initialize the Environment Isolation
```bash
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate # On Mac/Linux
```

### 2. Install Package Dependencies
```bash
pip install streamlit pypdf sentence-transformers chromadb numpy reportlab requests
```

### 3. Launch the Server Application
```bash
streamlit run app.py
```
Once initialized, navigate directly to `http://localhost:8501` to test the integrated auditing panel with the built-in testing asset generator.
