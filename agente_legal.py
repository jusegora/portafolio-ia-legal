import os
import json
import joblib
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def consultar_contrato(pregunta: str) -> str:
    try:
        modelo = SentenceTransformer('all-MiniLM-L6-v2')
        # Silenciamos la telemetría de ChromaDB
        cliente = chromadb.PersistentClient(path="data/chroma_db", settings=Settings(anonymized_telemetry=False))
        coleccion = cliente.get_collection(name="contratos_legales")
        
        vector = modelo.encode([pregunta]).tolist()
        resultados = coleccion.query(query_embeddings=vector, n_results=2)
        return "\n".join(resultados['documents'][0])
    except Exception as e:
        return f"Error al consultar contratos: {e}"

def predecir_resultado(tipo_caso: str, juez_id: str, monto: int) -> str:
    try:
        modelo = joblib.load('ml_predictivo/modelo_judicial.joblib')
        le_tipo = joblib.load('ml_predictivo/encoder_tipo.joblib')
        le_juez = joblib.load('ml_predictivo/encoder_juez.joblib')
        le_veredicto = joblib.load('ml_predictivo/encoder_veredicto.joblib')
        
        tipo_num = le_tipo.transform([tipo_caso])[0]
        juez_num = le_juez.transform([juez_id])[0]
        
        X = pd.DataFrame([[tipo_num, juez_num, monto]], columns=['tipo_caso_num', 'juez_num', 'monto_reclamado'])
        pred = modelo.predict(X)[0]
        prob = modelo.predict_proba(X)[0][1] * 100
        
        veredicto = le_veredicto.inverse_transform([pred])[0]
        return f"Veredicto pronosticado: {veredicto} con una probabilidad de exito del {prob:.2f}%."
    except Exception as e:
        return f"Error en el modelo predictivo: {e}"

herramientas = [
    {
        "type": "function",
        "function": {
            "name": "consultar_contrato",
            "description": "Usa esta herramienta cuando el usuario pregunte sobre clausulas, obligaciones, penalizaciones o terminos especificos de un contrato legal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pregunta": {"type": "string", "description": "La pregunta legal del usuario sobre el contrato"}
                },
                "required": ["pregunta"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "predecir_resultado",
            "description": "Usa esta herramienta cuando el usuario pregunte sobre la probabilidad de ganar un juicio o el resultado esperado de un caso.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tipo_caso": {"type": "string", "enum": ["Laboral", "Civil", "Penal", "Administrativo"], "description": "El tipo de caso legal"},
                    "juez_id": {"type": "string", "description": "El ID del juez asignado (ej. JUEZ-05)"},
                    "monto": {"type": "integer", "description": "El monto reclamado en dolares"}
                },
                "required": ["tipo_caso", "juez_id", "monto"]
            }
        }
    }
]

class AgenteLegal:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        self.llm = Groq(api_key=api_key, timeout=15.0)
        self.mensajes = [
            {"role": "system", "content": "Eres un abogado experto y asistente virtual. Tienes acceso a herramientas para consultar contratos y predecir resultados judiciales. Usa la herramienta adecuada segun la pregunta del usuario. Si no sabes algo, di que no tienes informacion."}
        ]

    def preguntar(self, texto: str):
        print(f"\nUsuario: {texto}")
        self.mensajes.append({"role": "user", "content": texto})
        
        respuesta = self.llm.chat.completions.create(
            messages=self.mensajes,
            model="openai/gpt-oss-20b",
            tools=herramientas,
            tool_choice="auto"
        )
        
        mensaje_respuesta = respuesta.choices[0].message
        self.mensajes.append(mensaje_respuesta)
        
        if mensaje_respuesta.tool_calls:
            for tool_call in mensaje_respuesta.tool_calls:
                nombre_funcion = tool_call.function.name
                argumentos = json.loads(tool_call.function.arguments)
                
                print(f"  [El Agente decidio usar la herramienta: {nombre_funcion}]")
                
                if nombre_funcion == "consultar_contrato":
                    resultado_herramienta = consultar_contrato(argumentos.get("pregunta"))
                elif nombre_funcion == "predecir_resultado":
                    resultado_herramienta = predecir_resultado(
                        argumentos.get("tipo_caso"), 
                        argumentos.get("juez_id"), 
                        argumentos.get("monto")
                    )
                else:
                    resultado_herramienta = "Herramienta no reconocida."
                    
                self.mensajes.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": nombre_funcion,
                    "content": str(resultado_herramienta)
                })
                
            respuesta_final = self.llm.chat.completions.create(
                messages=self.mensajes,
                model="openai/gpt-oss-20b"
            )
            print(f"\nAgente Legal: {respuesta_final.choices[0].message.content}")
            self.mensajes.append(respuesta_final.choices[0].message)
            
        else:
            print(f"\nAgente Legal: {mensaje_respuesta.content}")

if __name__ == "__main__":
    agente = AgenteLegal()
    agente.preguntar("¿De cuanto es la penalizacion si el contratista incumple?")
    agente.preguntar("Tengo un caso laboral con el JUEZ-05 por 30000 dolares. ¿Cual es mi probabilidad de ganar?")
