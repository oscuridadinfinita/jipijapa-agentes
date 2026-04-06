from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class VectorSearchToolInput(BaseModel):
    query: str = Field(..., description="Consulta en lenguaje natural sobre Jipijapa")

class VectorSearchTool(BaseTool):
    name: str = "Vector Search Tool"
    description: str = """
    Busca información en la base de conocimiento de Jipijapa.
    Útil para responder preguntas sobre historia, cultura, tradiciones, personajes,
    geografía, gastronomía y patrimonio del cantón Jipijapa.
    """
    args_schema: Type[BaseModel] = VectorSearchToolInput
    
    vectorstore: Optional[FAISS] = None
    
    def __init__(self, vectorstore: FAISS, **kwargs):
        super().__init__(**kwargs)
        self.vectorstore = vectorstore
    
    def _run(self, query: str) -> str:
        if not self.vectorstore:
            return "Error: Base de conocimiento no cargada."
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)
        
        if not docs:
            return "No se encontró información relevante."
        
        resultados = []
        for i, doc in enumerate(docs, 1):
            resultados.append(f"Fuente {i}:\n{doc.page_content[:500]}...\n")
        
        return "\n".join(resultados)
