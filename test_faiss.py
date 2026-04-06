from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Cargar variables de entorno desde .env
load_dotenv()

# Verificar que la clave se cargó (opcional, para depuración)
import os
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ ERROR: No se encontró OPENAI_API_KEY en .env")
    exit(1)
else:
    print(f"✅ API key cargada (primeros 10 chars): {api_key[:10]}...")

# Cargar la base guardada
vectorstore = FAISS.load_local("data/faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
print("✅ Base vectorial FAISS cargada correctamente")
print(f"Número de fragmentos: {vectorstore.index.ntotal}")
