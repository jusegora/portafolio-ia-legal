import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Cargamos las variables de entorno
load_dotenv()

class MotorRAGLegal:
    def __init__(self, ruta_db: str = "data/chroma_db"):
        print("Iniciando Motor RAG...")
        
        # 1. Motor de Embeddings (Búsqueda Vectorial)
        self.modelo_embeddings = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Desactivamos la telemetría para que no salgan errores en rojo
        self.cliente = chromadb.PersistentClient(
            path=ruta_db, 
            settings=Settings(anonymized_telemetry=False)
        )
        self.coleccion = self.cliente.get_or_create_collection(name="contratos_legales")
        
        # 2. Motor LLM (Generación con Groq)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Falta GROQ_API_KEY en el archivo .env")
            
        self.llm = Groq(api_key=api_key, timeout=10.0, max_retries=1)
        print("Sistemas listos.")

    def indexar_contrato(self, ruta_archivo: str):
        print(f"Indexando archivo: {ruta_archivo}")
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            texto = f.read()
            
        fragmentos = [p.strip() for p in texto.split('\n\n') if p.strip()]
        embeddings = self.modelo_embeddings.encode(fragmentos).tolist()
        
        ids = [f"clausula_{i}" for i in range(len(fragmentos))]
        metadatas = [{"fuente": ruta_archivo} for _ in fragmentos]
        
        self.coleccion.add(
            documents=fragmentos, 
            embeddings=embeddings, 
            metadatas=metadatas, 
            ids=ids
        )
        print(f"Indexación completada. {len(fragmentos)} fragmentos guardados.")

    def responder_pregunta(self, consulta: str):
        print(f"\nPregunta: {consulta}")
        
        vector_consulta = self.modelo_embeddings.encode([consulta]).tolist()
        resultados = self.coleccion.query(query_embeddings=vector_consulta, n_results=2)
        contexto = "\n".join(resultados['documents'][0])
        
        prompt_sistema = "Eres un abogado experto. Responde la pregunta del usuario utilizando UNICAMENTE el siguiente contexto extraido de un contrato. Si la respuesta no esta en el contexto, di que no tienes informacion suficiente."
        prompt_usuario = f"Contexto:\n{contexto}\n\nPregunta: {consulta}"
        
        print("Redactando respuesta legal con IA...")
        respuesta = self.llm.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": prompt_usuario}
            ],
            model="openai/gpt-oss-20b", 
            temperature=0.1 
        )
        
        return respuesta.choices[0].message.content

if __name__ == "__main__":
    motor = MotorRAGLegal()
    ruta_contrato = os.path.join("data", "contrato_prueba.txt")
    
    if motor.coleccion.count() == 0:
        motor.indexar_contrato(ruta_contrato)
    
    pregunta = "¿Qué sanciones economicas hay por incumplimiento?"
    respuesta_final = motor.responder_pregunta(pregunta)
    
    print("\n=== RESPUESTA DEL ABOGADO VIRTUAL ===")
    print(respuesta_final)
