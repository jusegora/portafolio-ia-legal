# Motor RAG Legal (Inteligencia Artificial)

Sistema de Búsqueda Semántica y Generación Aumentada por Recuperación (RAG) aplicado al sector legal.

## Descripcion
Este proyecto implementa un pipeline de IA que lee contratos en texto plano, los convierte en vectores matematicos (Embeddings) usando modelos Open Source, los almacena en una base de datos vectorial (ChromaDB) y utiliza un Modelo de Lenguaje (LLM via Groq) para responder preguntas complejas basadas estrictamente en el contexto del contrato.

## Tecnologias
*   **Python 3.11**
*   **Sentence-Transformers** (Modelo `all-MiniLM-L6-v2` para Embeddings)
*   **ChromaDB** (Base de datos vectorial)
*   **Groq API** (Modelo `openai/gpt-oss-20b` para generacion de texto)
*   **python-dotenv** (Gestion de credenciales)

## Como ejecutar
1. Clonar el repositorio.
2. Crear un entorno virtual e instalar dependencias: `pip install -r requirements.txt`
3. Crear un archivo `.env` con la clave `GROQ_API_KEY=tu_clave_aqui`.
4. Ejecutar el motor: `python src/rag_engine.py`

---
**Autor:** jusegora | Abogado & AI Solutions Architect | LegalTech & Análisis Forense de Datos
