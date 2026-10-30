
import os
import re

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


class DocumentProcessor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the local SentenceTransformer embedding model.
        """
        self.embedding_model = SentenceTransformer(model_name)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract readable text from all pages of a PDF.
        """
        reader = PdfReader(pdf_path)
        full_text = []

        for page in reader.pages:
            text = page.extract_text()

            if text:
                full_text.append(text)

        return "\n".join(full_text)

    def split_into_chunks(
        self,
        text: str,
        max_words: int = 100,
        overlap_sentences: int = 1
    ) -> list:
        """
        Split text into sentence-aware chunks.

        This avoids cutting sentences in the middle.
        Each chunk contains approximately max_words words.
        The final sentence of the previous chunk can be
        repeated to preserve context.
        """

        if not text or not text.strip():
            return []

        # Normalize unnecessary whitespace.
        text = re.sub(r"\s+", " ", text).strip()

        # Split text into sentences.
        sentences = re.split(
            r'(?<=[.!?])\s+',
            text
        )

        chunks = []
        current_sentences = []
        current_word_count = 0

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            words = sentence.split()
            word_count = len(words)

            # If adding the sentence exceeds the limit,
            # save the current chunk first.
            if (
                current_word_count + word_count > max_words
                and current_sentences
            ):
                chunks.append(
                    " ".join(current_sentences)
                )

                # Retain the last sentence for context overlap.
                if overlap_sentences > 0:
                    current_sentences = (
                        current_sentences[-overlap_sentences:]
                    )
                else:
                    current_sentences = []

                current_word_count = sum(
                    len(item.split())
                    for item in current_sentences
                )

            current_sentences.append(sentence)
            current_word_count += word_count

        # Save the final remaining chunk.
        if current_sentences:
            chunks.append(
                " ".join(current_sentences)
            )

        return chunks

    def process_document(self, pdf_path: str):
        """
        Extract PDF text, create sentence-aware chunks,
        generate embeddings, and prepare data for ChromaDB.
        """

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )

        # Step 1: Extract PDF text.
        raw_text = self.extract_text_from_pdf(pdf_path)

        if not raw_text.strip():
            return []

        # Step 2: Split into sentence-aware chunks.
        chunks = self.split_into_chunks(
            raw_text,
            max_words=100,
            overlap_sentences=1
        )

        if not chunks:
            return []

        # Step 3: Generate local embeddings.
        embeddings = self.embedding_model.encode(
            chunks,
            show_progress_bar=False
        )

        # Step 4: Prepare ChromaDB-compatible records.
        processed_data = []

        filename = os.path.basename(pdf_path)

        for idx, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):
            processed_data.append({
                "id": f"doc_{filename}_{idx}",
                "vector": embedding.tolist(),
                "metadata": {
                    "text": chunk,
                    "source": filename
                }
            })

        return processed_data
