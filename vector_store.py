import chromadb
from chromadb.config import Settings

class VectorIndex:
    def __init__(self):
        # Build an isolated local server instance in memory for fast performance
        self.client = chromadb.Client(Settings(allow_reset=True))
        self.collection = self.client.get_or_create_collection(name="compliance_index")

    def upsert_documents(self, processed_chunks):
        """Inserts vector arrays along with matching text metadata keys."""
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

    def query_context(self, query_vector: list, n_results: int = 2) -> list:
        """Retrieves matching text entries using cosine-similarity distances."""
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results
        )
        
        # Flatten ChromaDB's deeply nested output down into a standard list of strings
        flat_context = []
        if results and "documents" in results and results["documents"]:
            for nested_list in results["documents"]:
                for document in nested_list:
                    flat_context.append(document)
        return flat_context
