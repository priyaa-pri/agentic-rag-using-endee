"""
Endee Vector Store Integration
This module handles storing and retrieving embeddings using Endee
"""

from typing import List, Dict
import os

# Import Endee (installed from your fork)
from endee import EndeeClient


class EndeeVectorStore:
    def __init__(self, collection_name: str = "knowledge_base"):
        """
        Initialize Endee vector database
        """
        self.collection_name = collection_name
        self.client = EndeeClient()
        self.client.create_collection(collection_name)

    def add_documents(self, embeddings: List[List[float]], metadatas: List[Dict]):
        """
        Store document embeddings with metadata into Endee
        """
        for vector, metadata in zip(embeddings, metadatas):
            self.client.insert(
                collection_name=self.collection_name,
                vector=vector,
                metadata=metadata
            )

    def similarity_search(self, query_embedding: List[float], top_k: int = 3):
        """
        Perform vector similarity search using Endee
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            top_k=top_k
        )
        return results
