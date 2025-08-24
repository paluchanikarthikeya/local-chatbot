import os

UPLOAD_PATH = "upload"
DB_PATH = "vectorstore"

# Ensure upload folder exists
os.makedirs(UPLOAD_PATH, exist_ok=True)
import os
import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
import gradio as gr
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_PATH = "data"
DB_PATH = "vectorstore"

# Use HuggingFace embeddings (faster & no Ollama embedding warnings)
embeddings = HuggingFaceEmbeddings(model_name="C:/Users/reddy/OneDrive/Desktop/chatbot - Copy/all-MiniLM-L6-v2")

# Load existing vectorstore or create new
def get_vectorstore():
    if os.path.exists(DB_PATH):
        return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    else:
        return None

vectordb = get_vectorstore()
llm = Ollama(model="mistral")  # Use a small, fast model

qa_chain = None
if vectordb:
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Function to handle queries
def chatbot(query):
    global qa_chain
    if qa_chain is None:
        return "⚠️ Please upload a file (e.g. .pdf, .xls, .txt) before asking questions."
    
    result = qa_chain.invoke(query)  # Use .invoke instead of __call__
    answer = result["result"]
    sources = "\n".join([doc.metadata.get("source", "unknown") for doc in result["source_documents"]])
    return f"**Answer:** {answer}\n\n**Sources:**\n{sources if sources else 'No sources'}"
# Function to handle file uploads
def upload_file(file):
    global vectordb, qa_chain

    # Handle both file-like object and path string
    if hasattr(file, "read"):
        # file is a file-like object
        saved_path = os.path.join(UPLOAD_PATH, os.path.basename(getattr(file, "name", "uploaded_file")))
        with open(saved_path, "wb") as f:
            f.write(file.read())
    elif isinstance(file, str):
        # file is a path string
        saved_path = os.path.join(UPLOAD_PATH, os.path.basename(file))
        with open(file, "rb") as src, open(saved_path, "wb") as dst:
            dst.write(src.read())
    else:
        return "❌ Unsupported file type."

    # Load the file
    if saved_path.endswith(".pdf"):
        loader = PyPDFLoader(saved_path)
        docs = loader.load()
    else:
        loader = TextLoader(saved_path)
        docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)  # Smaller chunks for faster processing
    chunks = splitter.split_documents(docs)

    # If vector DB exists, add docs; otherwise create new one
    if vectordb is None:
        vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=DB_PATH)
    else:
        vectordb.add_documents(chunks)

    # Recreate QA chain with updated DB
    retriever = vectordb.as_retriever(search_kwargs={"k": 1})  # Retrieve fewer docs for speed
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever, 
        return_source_documents=True
    )

    return f"✅ File '{os.path.basename(saved_path)}' uploaded and indexed!"


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 📘 Local AI Q&A Bot (RAG with Ollama + LangChain)")
    
    with gr.Row():
        with gr.Column():
            query = gr.Textbox(label="Ask a question")
            submit = gr.Button("Ask")
        with gr.Column():
            output = gr.Markdown()

    with gr.Row():
        with gr.Column():
            file_upload = gr.File(label="Upload a document", file_types=[".pdf", ".txt", ".md"])
            upload_status = gr.Markdown()

    submit.click(fn=chatbot, inputs=query, outputs=output)
    file_upload.upload(fn=upload_file, inputs=file_upload, outputs=upload_status)

demo.launch()