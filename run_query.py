from dotenv import load_dotenv
from crewai import Crew
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from src.agents.researcher_agent import create_researcher_agent
from src.tools.vector_search_tool import VectorSearchTool
from src.tasks.answer_task import create_answer_task

load_dotenv()

def main():
    # Cargar la base vectorial FAISS
    print("📚 Cargando base de conocimiento...")
    vectorstore = FAISS.load_local(
        "data/faiss_index", 
        OpenAIEmbeddings(), 
        allow_dangerous_deserialization=True
    )
    print(f"✅ Base cargada: {vectorstore.index.ntotal} fragmentos")
    
    # Crear herramienta de búsqueda
    search_tool = VectorSearchTool(vectorstore=vectorstore)
    
    # Crear agente investigador
    researcher = create_researcher_agent(tools=[search_tool])
    
    # Pregunta de ejemplo
    question = "¿Quién fue Manuel Inocencio Parrales y Guale?"
    
    # Crear tarea
    task = create_answer_task(researcher, question, search_tool)
    
    # Crear crew y ejecutar
    crew = Crew(
        agents=[researcher],
        tasks=[task],
        verbose=True
    )
    
    result = crew.kickoff()
    print("\n" + "="*50)
    print("📝 RESPUESTA FINAL:")
    print("="*50)
    print(result)

if __name__ == "__main__":
    main()
