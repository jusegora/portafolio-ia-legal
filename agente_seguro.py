import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def redactar_demanda(tipo, monto, argumentos):
    print(f"\n[!] Alerta: Generando documento legal ({tipo} - ${monto})")
    
    texto = f"DEMANDA {tipo.upper()}\nMONTO: ${monto}\nHECHOS:\n{argumentos}"
    print("\n--- Borrador ---")
    print(texto)
    
    ok = input("\n¿Aprobar envio? (SI/NO): ").strip().upper()
    if ok == "SI":
        with open(f"demanda_{tipo.lower()}.txt", "w", encoding="utf-8") as f:
            f.write(texto)
        return {"status": "OK", "msg": "Documento guardado."}
    return {"status": "CANCELADO", "msg": "Cancelado por el usuario."}

herramientas = [
    {
        "type": "function",
        "function": {
            "name": "redactar_demanda",
            "description": "Genera una demanda legal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tipo": {"type": "string"},
                    "monto": {"type": "integer"},
                    "argumentos": {"type": "string"}
                },
                "required": ["tipo", "monto", "argumentos"]
            }
        }
    }
]

class AgenteSeguro:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=15.0)
        self.msgs = [{"role": "system", "content": "Eres un abogado. Usa la herramienta para redactar demandas."}]

    def run(self, texto):
        self.msgs.append({"role": "user", "content": texto})
        res = self.client.chat.completions.create(messages=self.msgs, model="openai/gpt-oss-20b", tools=herramientas, tool_choice="auto")
        msg = res.choices[0].message
        self.msgs.append(msg)
        
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                out = redactar_demanda(args.get("tipo"), args.get("monto"), args.get("argumentos"))
                
                if out["status"] == "CANCELADO":
                    print(f"\nAgente: Proceso detenido. {out['msg']}")
                    return
                
                self.msgs.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": out["msg"]})
                
            final = self.client.chat.completions.create(messages=self.msgs, model="openai/gpt-oss-20b")
            print(f"\nAgente: {final.choices[0].message.content}")

if __name__ == "__main__":
    a = AgenteSeguro()
    a.run("Redacta una demanda laboral por 50000 dolares por despido injustificado.")
