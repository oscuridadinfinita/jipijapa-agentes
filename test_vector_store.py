import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def create_vector_store(data_dir: str = "data/raw/", persist_dir: str = "data/faiss_index"):
    """Carga archivos .md, divide y guarda en FAISS"""
    documents = []
    for file in os.listdir(data_dir):
        if file.endswith(".md"):
            path = os.path.join(data_dir, file)
            loader = TextLoader(path, encoding="utf-8")
            documents.extend(loader.load())
            print(f"Cargado: {file}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"Se crearon {len(splits)} fragmentos (chunks).")
    
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    vectorstore.save_local(persist_dir)
    print(f"Base vectorial guardada en {persist_dir}")
    return vectorstore

if __name__ == "__main__":
    create_vector_store()
