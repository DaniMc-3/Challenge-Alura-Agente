import os
import sys
from dotenv import load_dotenv

# Cargar variables del archivo .env
load_dotenv(override=True)

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def obtener_api_key():
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        print(f"🔑 Clave Groq detectada: {api_key[:4]}...{api_key[-4:]}")
    
    if not api_key:
        raise ValueError(
            "❌ ERROR: No se encontró una GROQ_API_KEY válida en el archivo .env\n"
            "Asegúrate de incluir la línea: GROQ_API_KEY=gsk_... en tu archivo .env"
        )
    return api_key

def preparar_base_conocimiento():
    ruta_datos = "Documentos"
    
    if not os.path.exists(ruta_datos) or not os.listdir(ruta_datos):
        print(f"⚠️ Alerta: Coloca tus archivos PDF dentro de la carpeta '{ruta_datos}'.")
        return None

    print("📄 Cargando documentos PDF...", flush=True)
    loader = PyPDFDirectoryLoader(ruta_datos)
    documentos = loader.load()

    print(f"✂️ Fragmentando {len(documentos)} páginas de texto...", flush=True)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    fragmentos = text_splitter.split_documents(documentos)

    print("🧠 Indexando información localmente...", flush=True)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    base_conocimiento = Chroma.from_documents(fragmentos, embeddings)
    print("✅ ¡Base de conocimiento lista para usar!\n", flush=True)
    
    return base_conocimiento

def formato_documentos(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def crear_agente_soporte(base_conocimiento):
    api_key = obtener_api_key()
    
    # Modelo Llama 3.3 ejecutado en Groq
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0.2,
        groq_api_key=api_key
    )

    retriever = base_conocimiento.as_retriever(search_kwargs={"k": 3})

    system_prompt = (
        "Eres un agente de soporte de atención al cliente experto para nuestra Tienda Online Ferretería León.\n"
        "Tu objetivo es responder las preguntas de los usuarios utilizando ÚNICAMENTE la información "
        "proporcionada en el contexto de abajo (políticas de reembolso, envíos, privacidad y FAQs).\n\n"
        "Reglas estrictas:\n"
        "1. Si no encuentras la respuesta en el contexto, di amablemente: 'Lo siento, no tengo esa información detallada en mis políticas vigentes. Por favor, contacta a soporte@tienda.com'. No inventes información.\n"
        "2. Mantén un tono profesional, cortés y empático.\n"
        "3. Responde de forma clara y directa.\n\n"
        "Contexto extraído de las políticas:\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    cadena_rag = (
        {"context": retriever | formato_documentos, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return cadena_rag

if __name__ == "__main__":
    try:
        db = preparar_base_conocimiento()
        if db:
            agente = crear_agente_soporte(db)
            
            pregunta = "¿Cuáles son las políticas de devolución o garantías?"
            print(f"❓ Pregunta: {pregunta}\n", flush=True)
            
            respuesta = agente.invoke(pregunta)
            print(f"🤖 Respuesta:\n{respuesta}", flush=True)
    except Exception as e:
        print(f"\n❌ Ocurrió un error inesperado: {e}", file=sys.stderr)