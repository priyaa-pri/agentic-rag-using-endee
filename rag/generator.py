"""
RAG Answer Generator
- Takes retrieved context
- Generates a grounded answer
"""

class AnswerGenerator:
    def __init__(self):
        """
        Initialize generator
        (Rule-based now, can be extended to LLM later)
        """
        pass

    def generate_answer(self, query: str, retrieved_results):
        """
        Generate final answer using retrieved context
        """
        if not retrieved_results:
            return "No relevant information found in the knowledge base."

        context_snippets = []

        for result in retrieved_results:
            metadata = result.get("metadata", {})
            text = metadata.get("text", "")
            context_snippets.append(text)

        context = " ".join(context_snippets)

        answer = f"""
Question:
{query}

Answer (generated using retrieved knowledge):
{context}
"""
        return answer.strip()
