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

def buscar_contrato(pregunta):
    try:
        emb = SentenceTransformer('all-MiniLM-L6-v2')
        cli = chromadb.PersistentClient(path="data/chroma_db", settings=Settings(anonymized_telemetry=False))
        col = cli.get_collection(name="contratos_legales")
        res = col.query(query_embeddings=emb.encode([pregunta]).tolist(), n_results=2)
        return "\n".join(res['documents'][0])
    except Exception as e:
        return f"Error: {e}"

def predecir_caso(tipo, juez, monto):
    try:
        clf = joblib.load('ml_predictivo/modelo.joblib')
        le_t = joblib.load('ml_predictivo/enc_tipo.joblib')
        le_j = joblib.load('ml_predictivo/enc_juez.joblib')
        le_v = joblib.load('ml_predictivo/enc_verd.joblib')
        
        X = pd.DataFrame([[le_t.transform([tipo])[0], le_j.transform([juez])[0], monto]], 
                         columns=['tipo_num', 'juez_num', 'monto_reclamado'])
        
        pred = clf.predict(X)[0]
        prob = clf.predict_proba(X)[0][1] * 100
        return f"Veredicto: {le_v.inverse_transform([pred])[0]} ({prob:.2f}% exito)."
    except Exception as e:
        return f"Error: {e}"

herramientas = [
    {
        "type": "function",
        "function": {
            "name": "buscar_contrato",
            "description": "Busca informacion en contratos legales.",
            "parameters": {
                "type": "object",
                "properties": {"pregunta": {"type": "string"}},
                "required": ["pregunta"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "predecir_caso",
            "description": "Predice el resultado de un juicio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tipo": {"type": "string"},
                    "juez": {"type": "string"},
                    "monto": {"type": "integer"}
                },
                "required": ["tipo", "juez", "monto"]
            }
        }
    }
]

class Agente:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=15.0)
        self.msgs = [{"role": "system", "content": "Eres un abogado experto. Usa las herramientas si es necesario."}]

    def run(self, texto):
        print(f"\nUser: {texto}")
        self.msgs.append({"role": "user", "content": texto})
        
        res = self.client.chat.completions.create(
            messages=self.msgs, model="openai/gpt-oss-20b", tools=herramientas, tool_choice="auto"
        )
        
        msg = res.choices[0].message
        self.msgs.append(msg)
        
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                if tc.function.name == "buscar_contrato":
                    out = buscar_contrato(args.get("pregunta"))
                else:
                    out = predecir_caso(args.get("tipo"), args.get("juez"), args.get("monto"))
                    
                self.msgs.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": str(out)})
                
            final_res = self.client.chat.completions.create(messages=self.msgs, model="openai/gpt-oss-20b")
            print(f"Agente: {final_res.choices[0].message.content}")
        else:
            print(f"Agente: {msg.content}")

if __name__ == "__main__":
    a = Agente()
    a.run("¿De cuanto es la penalizacion por incumplimiento?")
    a.run("Probabilidad de ganar caso laboral con JUEZ-05 por 30000 dolares.")
