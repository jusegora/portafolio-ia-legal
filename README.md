# Proyectos de IA y LegalTech

Repositorio con herramientas de procesamiento de lenguaje natural, machine learning y agentes autonomos.

## Proyectos
- **RAG Engine (`src/rag_engine.py`)**: Busqueda semantica en contratos usando ChromaDB y Groq.
- **ML Predictivo (`ml_predictivo/`)**: Modelo Random Forest para predecir resultados judiciales.
- **Agente Autonomo (`agente_legal.py`)**: Agente que decide entre buscar en contratos o predecir juicios.
- **Agente Seguro (`agente_seguro.py`)**: Agente con aprobacion humana (HITL) para generar demandas.

## Setup
1. Crear entorno con Python 3.11.
2. Instalar dependencias: `pip install -r requirements.txt`
3. Crear archivo `.env` con `GROQ_API_KEY`.
