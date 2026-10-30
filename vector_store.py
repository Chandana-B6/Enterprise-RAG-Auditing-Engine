
import chromadb
from chromadb.config import Settings


class VectorIndex:
    def __init__(self):
        self.client = chromadb.Client(
            Settings(allow_reset=True)
        )

        self.collection = self.client.get_or_create_collection(
            name="compliance_index",
            metadata={"hnsw:space": "cosine"}
        )

    def upsert_documents(self, processed_chunks):
        if not processed_chunks:
            return

        ids = [item["id"] for item in processed_chunks]
        embeddings = [item["vector"] for item in processed_chunks]
        metadatas = [item["metadata"] for item in processed_chunks]
        documents = [item["metadata"]["text"] for item in processed_chunks]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

    def query_context(
        self,
        query_vector: list,
        n_results: int = 3,
        max_distance: float = 0.55
    ) -> list:

        count = self.collection.count()

        if count == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(n_results, count),
            include=["documents", "distances"]
        )

        documents = results.get("documents") or [[]]
        distances = results.get("distances") or [[]]

        relevant_chunks = []

        for document, distance in zip(
            documents[0],
            distances[0]
        ):
            if document and distance is not None:
                if distance <= max_distance:
                    relevant_chunks.append(document)

        return relevant_chunks
