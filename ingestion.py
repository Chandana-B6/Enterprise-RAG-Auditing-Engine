import os
from pypdf import PdfReader
# CORRECTED: Removed TransformerConfig to fix the ImportError
from sentence_transformers import SentenceTransformer

class DocumentProcessor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Utilizing SentenceTransformer (PyTorch backend) for local embeddings
        self.embedding_model = SentenceTransformer(model_name)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extracts raw text from a PDF file safely."""
        reader = PdfReader(pdf_path)
        full_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
        return "\n".join(full_text)

    def process_document(self, pdf_path: str):
        """Extracts text, creates fixed size chunks, and computes dense vectors."""
        raw_text = self.extract_text_from_pdf(pdf_path)
        
        # Native rolling-window chunking logic to avoid breaking library paths
        chunk_size = 500
        chunk_overlap = 50
        chunks = []
        for i in range(0, len(raw_text), chunk_size - chunk_overlap):
            segment = raw_text[i:i + chunk_size]
            if len(segment.strip()) > 10:  # Skip empty whitespace fragments
                chunks.append(segment)
        
        # Generate embeddings using local weights
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
        
        # Format payload structure explicitly for the Vector DB layer
        processed_data = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            processed_data.append({
                "id": f"doc_{os.path.basename(pdf_path)}_{idx}",
                "vector": embedding.tolist(),
                "metadata": {"text": chunk, "source": os.path.basename(pdf_path)}
            })
        return processed_data
