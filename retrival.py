
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

def filter_docs(docs, query):
    query = query.lower()
    
    # keep only chunks that contain key terms
    filtered = [
        doc for doc in docs 
        if any(word in doc.page_content.lower() for word in query.split())
    ]
    
    return filtered if filtered else docs


def retrival(querry):

    # setting model for embedding (getting the embedding model)
    embedding_model = embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ) 

    # setting the database
    db = Chroma(
        persist_directory="db",
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    retriver = db.as_retriever(search_kwargs={"k": 2}) # search 5 relevant one

    relevant_data = retriver.invoke(querry)

    # now we have the relevant data 
    print(relevant_data)

    return relevant_data



