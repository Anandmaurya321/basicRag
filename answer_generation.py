# Making the answer generation 

from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from retrival import retrival , filter_docs


# Load model ONCE (important for performance)
pipe = pipeline( "text2text-generation", model="google/flan-t5-base", max_new_tokens=256 )

model = HuggingFacePipeline(pipeline=pipe)


def answer_generation(query):
    
    docs = retrival(query)
    docs = filter_docs(docs , query)

    context = "\n\n".join([
    f"[Chunk {i+1}]\n{doc.page_content.strip()}"
    for i, doc in enumerate(docs)
    ])
    
    combined_input = f"""
    Answer the question clearly in 1-2 sentences.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    result = model.invoke(combined_input)

    return result


def main():
    query = input("give your query here: ")
    print(answer_generation(query))


if __name__ == "__main__":
    main()


    