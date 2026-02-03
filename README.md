# Agentic RAG using Endee Vector Database
## Overview
This project implements an Agentic Retrieval-Augmented Generation (RAG) system using Endee as the vector database. The goal of the project is to enable semantic question answering over a small knowledge base such as internship rules or evaluation policies. Instead of relying on keyword matching, the system uses embeddings and vector similarity search to retrieve relevant information. This project was developed as part of a project-based evaluation to demonstrate practical usage of Endee in an AI/ML workflow.
## Why This Project
While learning about vector databases and RAG systems, I wanted to move beyond a basic document search or chatbot example. This project focuses on using Endee as the core vector store, implementing semantic similarity search, building a simple agent-driven retrieval pipeline, and keeping the system modular and easy to extend.
## What the System Does
The system loads text documents from a local knowledge base, converts the content into embeddings, stores the embeddings and metadata in Endee, takes a user question and converts it into an embedding, retrieves the most relevant content using vector similarity search, and generates an answer grounded in the retrieved data.
## Architecture
User Question → Query Embedding → Endee Vector Search → Relevant Context → Answer Generation (RAG)
## Technologies Used
Python, Endee (Vector Database), Sentence Transformers, Semantic Search, Retrieval-Augmented Generation (RAG), Agent-based retrieval logic
## Project Structure
agentic-rag-using-endee/
├── agent/
│   ├── intent_analyzer.py
│   └── retrieval_agent.py
├── data/
│   └── knowledge_docs/
│       ├── attendance_policy.txt
│       ├── evaluation_process.txt
│       └── internship_rules.txt
├── embeddings/
│   └── embedder.py
├── rag/
│   └── generator.py
├── vector_store/
│   └── endee_store.py
├── app.py
├── requirements.txt
└── README.md
## How the Pipeline Works
Knowledge Ingestion: Text files inside data/knowledge_docs are loaded, split into smaller chunks, converted into embeddings using a sentence transformer model, and stored in Endee along with metadata.
Agentic Retrieval: When a user asks a question, the query is converted into an embedding, Endee performs vector similarity search, and the most relevant document chunks are retrieved.
Answer Generation: The retrieved context is combined and used to generate a final answer that is grounded only in the stored knowledge base.
## Setup Instructions
Clone the repository:
git clone https://github.com/priyaa-pri/agentic-rag-using-endee.git
cd agentic-rag-using-endee
Create and activate virtual environment:
python -m venv venv
venv\Scripts\activate
Install dependencies:
pip install -r requirements.txt
Run the application:
python app.py
## Example
Ask a question (or type 'exit'): What happens if attendance is irregular?
The system retrieves relevant rules from the knowledge base and returns a context-aware answer.
## Possible Improvements
Integrating an LLM for more natural responses, adding multi-step agent reasoning, supporting dynamic document uploads, and building a web interface using Streamlit or Flask.
## Author
Priya R  
B.E – Artificial Intelligence & Data Science
## Notes
This project was built to understand and demonstrate how vector databases like Endee can be used in practical AI applications involving semantic search and RAG.

