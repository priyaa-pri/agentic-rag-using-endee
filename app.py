"""
Main Application Entry
Agentic RAG using Endee Vector Database
"""

from agent.retrieval_agent import RetrievalAgent
from rag.generator import AnswerGenerator


def main():
    print("Initializing Agentic RAG System using Endee...\n")

    agent = RetrievalAgent()
    generator = AnswerGenerator()

    # Step 1: Ingest knowledge base
    print("Ingesting knowledge base...")
    agent.ingest_knowledge_base("data/knowledge_docs")
    print("Knowledge base ingested successfully.\n")

    # Step 2: User query loop
    while True:
        query = input("Ask a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        results = agent.retrieve_context(query)
        answer = generator.generate_answer(query, results)

        print("\n" + "=" * 50)
        print(answer)
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
