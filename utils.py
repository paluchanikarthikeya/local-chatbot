from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

DB_PATH = "vectorstore"

def get_vectorstore():
    embeddings = OllamaEmbeddings(model="mistral")
    vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    return vectordb
