import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Paths
DATA_PATH = "data"
DB_PATH = "vectorstore"

def load_documents():
    """Load all supported documents from data/ folder."""
    docs = []
    for file in os.listdir(DATA_PATH):
        path = os.path.join(DATA_PATH, file)
        if not os.path.isfile(path):
            continue
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        elif file.endswith(".txt") or file.endswith(".md"):
            loader = TextLoader(path)
            docs.extend(loader.load())
    return docs

def create_vector_store():
    documents = load_documents()
    if not documents:
        print("⚠️ No documents found in 'data/'. Please add some PDFs, TXT, or MD files.")
        return

    # Split docs into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"📑 Loaded {len(documents)} documents → split into {len(chunks)} chunks.")

    # Use HuggingFace embeddings (lightweight & CPU-friendly)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create Chroma vectorstore
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=DB_PATH)
    vectordb.persist()

    print(f"✅ Vector DB created & saved at '{DB_PATH}'")

if __name__ == "__main__":
    create_vector_store()
