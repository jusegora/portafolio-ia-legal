# Portafolio de IA, LegalTech y Legal Operations

**Autor:** jusegora  
*Abogado | AI Solutions Architect | LegalTech & Data Science*

Este repositorio contiene una colección de sistemas de Inteligencia Artificial y Analítica Avanzada aplicados al sector legal, demostrando la integración de desarrollo de software, ciencia de datos y visión de negocio.

---

## 🏗️ Proyecto 1: Motor RAG Legal (Búsqueda Semántica)
*   **Ubicación:** `src/`
*   **Descripción:** Pipeline RAG (Retrieval-Augmented Generation) que lee contratos en texto plano, los convierte en vectores matemáticos (Embeddings) y los almacena en una base de datos vectorial (ChromaDB). Utiliza un LLM (Groq) para responder preguntas legales basadas estrictamente en el contexto del documento.
*   **Tecnologías:** Python, Sentence-Transformers, ChromaDB, Groq API.

## 🏗️ Proyecto 2: Analítica Predictiva Judicial (Machine Learning)
*   **Ubicación:** `ml_predictivo/`
*   **Descripción:** Modelo de Machine Learning (Random Forest) entrenado con 5,000 casos judiciales sintéticos. Predice la probabilidad de éxito de una demanda basándose en el tipo de caso, el juez y el monto reclamado.
*   **Tecnologías:** Pandas, Scikit-Learn, Joblib.

## 🏗️ Proyecto 3: Agente Legal Autónomo (Agentic AI)
*   **Ubicación:** `agente_legal.py`
*   **Descripción:** Agente de IA con arquitectura de *Function Calling*. Evalúa la pregunta del usuario y decide de forma autónoma si debe usar la herramienta de búsqueda semántica (RAG) o el modelo predictivo (ML).
*   **Tecnologías:** Groq API, JSON Tool Calling.

## 🏗️ Proyecto 4: Sistema Seguro con Human-in-the-Loop (Gobernanza)
*   **Ubicación:** `agente_seguro.py`
*   **Descripción:** Implementación de seguridad de nivel Enterprise. El agente puede redactar demandas, pero el sistema se detiene y exige aprobación humana explícita (SI/NO) antes de ejecutar acciones irreversibles. Incluye *Guardrails* a nivel de código Python.
*   **Tecnologías:** Python, Groq API.

## 🏗️ Proyecto 5: Dashboard de Legal Operations (Power BI & DAX)
*   **Ubicación:** `LegalOps_Dashboard.pbix` y `LegalOps_Dashboard.pdf`
*   **Descripción:** Dashboard ejecutivo interactivo para la toma de decisiones legales. Incluye visualizaciones de veredictos, distribución de casos y una métrica DAX personalizada para calcular el "Monto en Riesgo" económico.
*   **Tecnologías:** Microsoft Power BI, DAX.
