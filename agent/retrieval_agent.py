"""
Agentic Retrieval Module
- Analyzes user query
- Retrieves relevant context from Endee
- Acts as a decision-making retrieval agent
"""

from embeddings.embedder import DocumentEmbedder
from vector_store.endee_store import EndeeVectorStore


class RetrievalAgent:
    def __init__(self):
        """
        Initialize retrieval agent with embedder and vector store
        """
        self.embedder = DocumentEmbedder()
        self.vector_store = EndeeVectorStore()

    def ingest_knowledge_base(self, data_dir: str):
        """
        Load documents, generate embeddings, and store them in Endee
        """
        documents = self.embedder.load_documents(data_dir)
        embeddings, metadatas = self.embedder.generate_embeddings(documents)

        self.vector_store.add_documents(embeddings, metadatas)

    def retrieve_context(self, query: str, top_k: int = 3):
        """
        Retrieve relevant context for a user query
        """
        # Convert query to embedding
        query_embedding = self.embedder.model.encode(query).tolist()

        # Search using Endee vector DB
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        return results
