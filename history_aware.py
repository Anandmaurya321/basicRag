from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from answer_generation import answer_generation


# Load model once
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256
)

model = HuggingFacePipeline(pipeline=pipe)

history = []


def askQuestion(query):
    
    # Step 1: Convert follow-up → standalone question
    if len(history) != 0:

        history_text = "\n".join(history)

        prompt = f"""
        Given the chat history and a new question, rewrite it as a standalone question.

        Chat History:
        {history_text}

        New Question:
        {query}
        """

        userQuestion = model.invoke(prompt)

    else:
        userQuestion = query

    # Step 2: Get answer using your RAG pipeline
    res = answer_generation(userQuestion)

    # Step 3: Store history (simple text instead of messages)
    history.append(f"User: {userQuestion}")
    history.append(f"AI: {res}")

    return res


def startfun():
    while True:
        query = input("Give your query here: ")

        if query.lower() == 'quit':
            print("goodbye !")
            return
        
        print(askQuestion(query))


def main():
    startfun()


if __name__ == "__main__":
    main()