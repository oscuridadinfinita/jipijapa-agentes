from crewai import Task

def create_answer_task(agent, question: str, search_tool):
    """Crea una tarea para responder preguntas sobre Jipijapa"""
    return Task(
        description=f"""
        Responde la siguiente pregunta sobre la cultura, historia o tradiciones de Jipijapa:
        
        Pregunta: {question}
        
        Usa la herramienta de búsqueda vectorial para encontrar información relevante en la base de conocimiento.
        Si no encuentras información suficiente, indica amablemente que no tienes datos sobre ese tema específico.
        """,
        expected_output="Una respuesta clara y precisa en español, con referencias a las fuentes si es posible.",
        agent=agent,
        tools=[search_tool]
    )
