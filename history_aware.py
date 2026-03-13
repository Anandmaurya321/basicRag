
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
from answer_generation import answer_generation

from dotenv import load_dotenv
load_dotenv()


model = ChatOpenAI(model="gpt-4o")

history = []

def askQuestion(querry):
    
    userQuestion=""

    if(len(history)!=0):

        messages = [
          SystemMessage(content="Given the chat History based on it rewrite the new question standalone. just return the standalone new question")
        ]
        + history + [
            HumanMessage(content="New Question : {querry}")
            ]
        
        userQuestion = model.invoke(messages)
   
    else:
        userQuestion = querry
    
    # now we have a standalone user question

    res = answer_generation(userQuestion)

    history.append(SystemMessage(content=userQuestion))
    history.append(AIMessage(content=res))

    return res


def startfun():
    while(True):
        querry = input("Give your querry here : ")
        if(querry.lower()=='quit'):
            print("goodbye !")
            return 
        
        else:
            print(askQuestion(querry))
        


def main():
    startfun()


if __name__ == "__main__":
    main()




        