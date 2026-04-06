import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from crewai import Agent, Task, Crew
import os

# Cargar variables de entorno
load_dotenv()

# Configurar la página
st.set_page_config(
    page_title="Agente Cultural de Jipijapa",
    page_icon="🌿",
    layout="centered"
)

# Título y descripción
st.title("🌿 Agente Cultural de Jipijapa")
st.markdown("""
*Un asistente inteligente que responde preguntas sobre la historia, cultura, 
tradiciones y patrimonio del cantón Jipijapa.*
""")

# Sidebar con información
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e8/Flag_of_Manab%C3%AD.svg", width=100)
    st.markdown("## 📖 Sobre el proyecto")
    st.markdown("""
    Este agente fue entrenado con documentos históricos y culturales de Jipijapa, incluyendo:
    - Ciencias ambientales y geografía
    - Antropología y genética de poblaciones
    - Arqueología e ingeniería prehispánica
    - Lingüística y toponimia
    - Demografía histórica
    - Historia económica y comercio
    - Sociología política y litigios
    - Etnografía contemporánea
    """)
    st.markdown("---")
    st.markdown("🤝 **Asociación MOMOTUS**")
    st.markdown("Turismo Comunitario en Jipijapa")

# Inicializar la base de conocimiento
@st.cache_resource
def load_vectorstore():
    """Carga la base vectorial FAISS (se cachea para no recargar cada vez)"""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        "data/faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    return vectorstore

# Crear herramienta de búsqueda personalizada
class VectorSearchTool:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
    
    def run(self, query: str) -> str:
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)
        if not docs:
            return "No se encontró información relevante sobre esa consulta."
        resultados = []
        for i, doc in enumerate(docs, 1):
            resultados.append(f"**Fuente {i}:** {doc.page_content[:500]}...")
        return "\n\n".join(resultados)

# Crear el agente
def create_agent(search_tool):
    return Agent(
        role="Investigador Cultural de Jipijapa",
        goal="Proporcionar información precisa y detallada sobre la historia, cultura, geografía y tradiciones del cantón Jipijapa",
        backstory="""Eres un etnohistoriador experto en la cultura de Jipijapa. 
        Has estudiado los pozos ancestrales, el sombrero de paja toquilla, 
        la gastronomía patrimonial y las luchas indígenas de la región. 
        Respondes con precisión, citando fuentes cuando es posible, y con un tono cálido y respetuoso.""",
        verbose=False,
        allow_delegation=False
    )

# Interfaz de preguntas
st.markdown("---")
st.markdown("## 💬 Haz una pregunta")

# Input del usuario
question = st.text_input("Escribe tu pregunta aquí:", placeholder="Ejemplo: ¿Quién fue Manuel Inocencio Parrales y Guale?")

if st.button("🔍 Preguntar", type="primary"):
    if question:
        with st.spinner("Consultando la sabiduría de Jipijapa..."):
            try:
                # Cargar la base vectorial
                vectorstore = load_vectorstore()
                
                # Crear herramienta de búsqueda
                search_tool = VectorSearchTool(vectorstore)
                
                # Crear agente
                agent = create_agent(search_tool)
                
                # Crear tarea
                task = Task(
                    description=f"""
                    Responde la siguiente pregunta sobre la cultura, historia o tradiciones de Jipijapa:
                    
                    Pregunta: {question}
                    
                    Usa la herramienta de búsqueda vectorial para encontrar información relevante.
                    Si no encuentras información suficiente, indica amablemente que no tienes datos sobre ese tema específico.
                    """,
                    expected_output="Una respuesta clara y precisa en español, con referencias a las fuentes si es posible.",
                    agent=agent,
                    tools=[search_tool]
                )
                
                # Crear crew y ejecutar
                crew = Crew(agents=[agent], tasks=[task], verbose=False)
                result = crew.kickoff()
                
                # Mostrar respuesta
                st.markdown("### 📝 Respuesta:")
                st.success(result)
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.markdown("""
                ⚠️ **Posibles causas:**
                - La API key de OpenAI no está configurada en Streamlit Cloud.
                - La base de conocimiento no se cargó correctamente.
                """)
    else:
        st.warning("Por favor, escribe una pregunta.")

# Footer
st.markdown("---")
st.markdown(
    "<center><small>Desarrollado con ❤️ para la Asociación MOMOTUS y la cultura de Jipijapa</small></center>",
    unsafe_allow_html=True
)
