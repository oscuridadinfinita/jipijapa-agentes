import os
from pathlib import Path
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_markdown_documents(raw_dir: str = "data/raw"):
    """Carga todos los archivos .md del directorio raw y devuelve lista de documentos (con metadata de sección)."""
    from langchain.schema import Document
    docs = []
    for file_path in Path(raw_dir).glob("*.md"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        # Añadir metadato con el nombre del archivo (útil para filtrado)
        metadata = {"source": file_path.name}
        docs.append(Document(page_content=text, metadata=metadata))
    return docs

def split_documents(docs):
    """Divide documentos Markdown respetando encabezados (hasta nivel 2) y luego chunking adicional."""
    # Primero, dividir por encabezados Markdown (hasta ##)
    headers_to_split_on = [
        ("#", "Header1"),
        ("##", "Header2"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    all_chunks = []
    for doc in docs:
        # El splitter trabaja sobre texto plano, no sobre Document
        splits = markdown_splitter.split_text(doc.page_content)
        for split in splits:
            # Fusionar metadatos: los headers se añaden automáticamente al metadata del split
            split.metadata.update(doc.metadata)
            all_chunks.append(split)
    
    # Opcional: chunking recursivo adicional para trozos más pequeños (ej. si algún fragmento aún es largo)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    final_chunks = text_splitter.split_documents(all_chunks)
    return final_chunks

def create_vector_store(chunks, persist_directory="data/vector_store"):
    """Crea embeddings y guarda en Chroma. Retorna el vectorstore."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # o "text-embedding-ada-002"
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    print(f"Vector store creado con {vectorstore._collection.count()} chunks en {persist_directory}")
    return vectorstore

if __name__ == "__main__":
    # Prueba rápida
    docs = load_markdown_documents()
    print(f"Cargados {len(docs)} documentos")
    chunks = split_documents(docs)
    print(f"Generados {len(chunks)} chunks")
    vectorstore = create_vector_store(chunks)
