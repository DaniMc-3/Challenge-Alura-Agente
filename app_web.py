import os
import streamlit as st
from dotenv import load_dotenv

# Cargar variables de entorno (.env)
load_dotenv(override=True)

# Importar las funciones de tu APP.py
from APP import preparar_base_conocimiento, crear_agente_soporte

# Configuración de la página web
st.set_page_config(
    page_title="Soporte - Ferretería León",
    page_icon="🛠️",
    layout="centered"
)

# Título y Encabezado
st.title("🛠️ Ferretería León")
st.subheader("Asistente Virtual de Soporte y Políticas")
st.markdown("---")

# Cargar la base de conocimiento usando el caché de Streamlit
@st.cache_resource
def inicializar_agente():
    db = preparar_base_conocimiento()
    if db:
        return crear_agente_soporte(db)
    return None

agente = inicializar_agente()

if not agente:
    st.error("⚠️ No se pudo cargar la base de conocimiento. Revisa la carpeta 'Documentos'.")
    st.stop()

# Historial del chat en la sesión
if "mensajes" not in st.session_state:
    st.session_state.mensajes = [
        {"role": "assistant", "content": "¡Hola! 👋 Bienvenido a Ferretería León. ¿En qué puedo ayudarte hoy sobre nuestras políticas, envíos o devoluciones?"}
    ]

# Mostrar historial de mensajes
for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])

# Entrada de texto para el usuario
if pregunta := st.chat_input("Escribe tu consulta aquí..."):
    # Guardar y mostrar mensaje del usuario
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    # Generar respuesta del agente
    with st.chat_message("assistant"):
        with st.spinner("Consultando políticas..."):
            try:
                respuesta = agente.invoke(pregunta)
                st.markdown(respuesta)
                st.session_state.mensajes.append({"role": "assistant", "content": respuesta})
            except Exception as e:
                st.error(f"❌ Ocurrió un error: {e}")
