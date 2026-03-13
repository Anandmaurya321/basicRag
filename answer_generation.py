
# Making the answer generation 

from retrival import retrival
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

querry = "what is google"

combined_input = "based on the following document answer this question: {querry}"

model = ChatOpenAI(model="gpt-4o")

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content=combined_input)
]


result = model.invoke(messages)

print(result)

