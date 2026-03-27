
from answer_generation import answer_generation

def get_answer(query: str) -> str:
    """
    Simple stateless RAG function
    Input: user query
    Output: generated answer
    """
    
    if not query or query.strip() == "":
        return "Please provide a valid question."

    try:
        response = answer_generation(query)
        return response
    except Exception as e:
        return f"Error generating answer: {str(e)}"