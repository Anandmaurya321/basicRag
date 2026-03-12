import os # operating system
from langchain_community.document_loaders import TextLoader, DirectoryLoader # for loading directory , # for loading text file 
from langchain_text_splitters import CharacterTextSplitter # for chunking : using Character-Text-Splitter
from langchain_huggingface import HuggingFaceEmbeddings # for creating embeddings for chunks : converting chunks into vector
from langchain_chroma import Chroma # database use for storing the vectors : Special Database
from dotenv import load_dotenv # for loading env file 

load_dotenv()


# 1. upload the documents

def upload_documents(path = "docs"):

    if not os.path.exists(path):
        raise FileNotFoundError(f"The directory {path} does not exist. Please create it and add your company file so that we can take it.")
    
    # Here we are loading the whole directory :
    # We have the directory path '/docs' and we just going to load the whole directory
    # using DirectoryLoader by defining the requried file extension

    loader = DirectoryLoader(
        path= path,   # directory path : which you want to load
        glob="*.txt", # required files to load 
        loader_cls=TextLoader, 
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load() # will load in the documents

    if len(documents)==0:
        raise FileNotFoundError(f"No .txt files found in {path}. Please add your company documents so that we can take it.")
    
    print("uploading done")

    return documents


# 2. Chunking the documents with required text splitter
def chunking(documents):
    text_spliter = CharacterTextSplitter(
        chunk_size=100, 
        chunk_overlap=0
    )

    chunks = text_spliter.split_documents(documents)
    print("chunking done")
    return chunks


# 3. Creating Embeddings for that chunks and save into vector

def embedding_and_storing(chunks , db_path = "db"):
    embedding_model = embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )# taking the embedding model 

    # creating embeddings and storing inside chroma db (vector db)

    vectorstore = Chroma.from_documents(
        documents=chunks,  # passing chunks 
        embedding=embedding_model, # using embedding model 
        persist_directory=db_path,  # where to store it 
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("vector store done")

    return vectorstore

def main():
    documents = upload_documents("docs")

    chunks = chunking(documents)

    vector_store = embedding_and_storing(chunks , "db")

    return vector_store

if __name__ == "__main__":
    main()