# Making the answer generation 

from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline


# Load model ONCE (important for performance)
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256
)

model = HuggingFacePipeline(pipeline=pipe)


def answer_generation(query):
    combined_input = f"Based on the following document answer this question: {query}"

    result = model.invoke(combined_input)

    return result


def main():
    query = input("give your query here: ")
    print(answer_generation(query))


if __name__ == "__main__":
    main()