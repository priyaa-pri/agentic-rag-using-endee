"""
Embedding generation module
- Reads knowledge base documents
- Splits text into chunks
- Generates embeddings
"""

import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer


class DocumentEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        """
        self.model = SentenceTransformer(model_name)

    def load_documents(self, data_dir: str) -> List[Dict]:
        """
        Load text documents from data directory
        """
        documents = []

        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(data_dir, filename)

                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                documents.append({
                    "content": text,
                    "source": filename
                })

        return documents

    def chunk_text(self, text: str, chunk_size: int = 200) -> List[str]:
        """
        Split text into smaller chunks
        """
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)

        return chunks

    def generate_embeddings(self, documents: List[Dict]):
        """
        Generate embeddings and metadata
        """
        embeddings = []
        metadatas = []

        for doc in documents:
            chunks = self.chunk_text(doc["content"])

            for idx, chunk in enumerate(chunks):
                vector = self.model.encode(chunk).tolist()

                embeddings.append(vector)
                metadatas.append({
                    "source": doc["source"],
                    "chunk_id": idx,
                    "text": chunk
                })

        return embeddings, metadatas
