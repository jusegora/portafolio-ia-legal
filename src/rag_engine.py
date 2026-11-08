import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class MotorRAGLegal:
    def __init__(self, ruta_db: str = "data/chroma_db"):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.cliente = chromadb.PersistentClient(
            path=ruta_db, 
            settings=Settings(anonymized_telemetry=False)
        )
        self.coleccion = self.cliente.get_or_create_collection(name="contratos_legales")
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Falta GROQ_API_KEY en el .env")
            
        self.llm = Groq(api_key=api_key, timeout=10.0, max_retries=1)

    def indexar(self, ruta_archivo: str):
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            texto = f.read()
            
        fragmentos = [p.strip() for p in texto.split('\n\n') if p.strip()]
        vectores = self.embedder.encode(fragmentos).tolist()
        
        ids = [f"clausula_{i}" for i in range(len(fragmentos))]
        metadatas = [{"fuente": ruta_archivo} for _ in fragmentos]
        
        self.coleccion.add(documents=fragmentos, embeddings=vectores, metadatas=metadatas, ids=ids)
        print(f"Indexados {len(fragmentos)} fragmentos.")

    def consultar(self, pregunta: str):
        vector_pregunta = self.embedder.encode([pregunta]).tolist()
        resultados = self.coleccion.query(query_embeddings=vector_pregunta, n_results=2)
        contexto = "\n".join(resultados['documents'][0])
        
        prompt = f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"
        
        res = self.llm.chat.completions.create(
            messages=[
                {"role": "system", "content": "Eres un abogado. Responde usando solo el contexto provisto."},
                {"role": "user", "content": prompt}
            ],
            model="openai/gpt-oss-20b", 
            temperature=0.1 
        )
        return res.choices[0].message.content

if __name__ == "__main__":
    motor = MotorRAGLegal()
    ruta = os.path.join("data", "contrato_prueba.txt")
    
    if motor.coleccion.count() == 0:
        motor.indexar(ruta)
    
    print(motor.consultar("¿Qué sanciones hay por incumplimiento?"))
