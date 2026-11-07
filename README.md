# Portafolio de IA y LegalTech

Colección de sistemas de Inteligencia Artificial aplicados al sector legal.

## Proyecto 1: Motor RAG Legal (Búsqueda Semántica)
*   **Ubicacion:** `src/`
*   **Descripcion:** Pipeline RAG que convierte contratos en vectores, los almacena en ChromaDB y usa un LLM (Groq) para responder preguntas basadas en el contexto.

## Proyecto 2: Analitica Predictiva Judicial (Machine Learning)
*   **Ubicacion:** `ml_predictivo/`
*   **Descripcion:** Modelo de Machine Learning (Random Forest) entrenado con 5,000 casos sinteticos. Predice la probabilidad de exito de una demanda.
*   **Tecnologias:** Pandas, Scikit-Learn, Joblib.

## Proyecto 3: Agente Legal Autonomo (Agentic AI)
*   **Ubicacion:** `agente_legal.py`
*   **Descripcion:** Agente de IA con arquitectura de Function Calling. Decide de forma autonoma si debe usar la busqueda vectorial (RAG) o el modelo predictivo (ML) dependiendo de la pregunta del usuario.

---
**Autor:** jusegora
