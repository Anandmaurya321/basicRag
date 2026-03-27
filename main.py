from fastapi import FastAPI
from pydantic import BaseModel
from final_rag import get_answer

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/")
def ask(query: Query):
    answer = get_answer(query.question)
    return {"answer": answer}

