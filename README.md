#  Agente de IA para Soporte al Cliente - Ferretería León

Este proyecto consiste en un **Agente Virtual de Atención al Cliente y Soporte** desarrollado para el **Challenge Alura Latam – Agentes de Inteligencia Artificial**. 

El sistema utiliza la técnica de **Retrieval-Augmented Generation (RAG)** para responder consultas sobre políticas de devoluciones, garantías, envíos y privacidad de la **Ferretería León**, utilizando únicamente la documentación interna y garantizando respuestas precisas sin alucinaciones.

---

##  Características Principales

* **Arquitectura RAG:** Procesa documentos en PDF para responder con precisión basándose en contexto real.
* **Procesamiento de Lenguaje Natural:** Potenciado por **Llama 3.3 (70B)** a través de la API ultrarrápida de **Groq**.
* **Base de Datos Vectorial Local:** Utiliza `ChromaDB` y embeddings de HuggingFace (`all-MiniLM-L6-v2`) para búsqueda semántica eficiente.
* **Interfaz Web Intuitiva:** Implementada en **Streamlit** para ofrecer un chat interactivo e instantáneo.
* **Resguardo de Información:** Respuestas estrictamente delimitadas por las políticas vigentes.

---

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python 3.10+
* **Framework IA:** LangChain
* **LLM:** Meta Llama 3.3 70B (vía `langchain-groq`)
* **Vector Store:** ChromaDB
* **Embeddings:** HuggingFace Embeddings
* **Interfaz Web:** Streamlit
* **Gestión de Entorno:** `python-dotenv`
* **Oracle Cloud Infrastructure (OCI)

---

## 📋 Requisitos Previos

1. Tener instalado **Python 3.10** o superior.
2. Contar con una API Key de **Groq** (se obtiene gratuitamente en [console.groq.com](https://console.groq.com)).

---

## ⚙️ Instalación y Configuración

1. **Clonar el repositorio:**
   ```bash
   git clone <URL_DE_TU_REPOSITO_EN_GITHUB>
   cd "Agente Alura IA"

## ⚙️ Instalación y Uso Rápido

1. **Clonar el proyecto y entrar a la carpeta:**
   ```bash
   git clone <URL_DE_TU_REPOSITO_EN_GITHUB>
   cd "Agente Alura IA"
Instalacion de dependencias:

Bash
pip install langchain-groq langchain-community langchain-huggingface chromadb streamlit python-dotenv pypdf
Configurar tu clave API:
Creamos un archivo llamado .env en la carpeta del proyecto y agrega tu clave de Groq:

Fragmento de código
GROQ_API_KEY="gsk_tu_clave_aqui"
Lanzar la aplicación Web:

Bash
streamlit run app_web.py

## 📂 Estructura del Proyecto
Plaintext
Agente Alura IA/

├── Documentos/          # Nuestros PDFs con políticas y FAQs

├── .env                 # Nuestra clave GROQ_API_KEY

├── APP.py               # Lógica del agente RAG

├── app_web.py           # Interfaz gráfica en web

└── README.md            # Documentación del proyecto


Proyecto desarrollado para el Challenge Alura Latam.


---

### Pasos finales para guardar todo lo que hicimos en Git:

1. Reemplaza el texto en tu archivo **`README.md`** y guárdalo (`Ctrl + S`).
2. En tu terminal ejecuta:
   ```powershell
   git add README.md
   git commit -m "docs: Simplificada seccion de instalacion en README"
   git push   
