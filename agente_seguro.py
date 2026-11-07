import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def redactar_demanda_borrador(tipo_caso: str, monto: int, argumentos: str):
    print("\n" + "="*50)
    print("⚠️  ALERTA DE GOBERNANZA: ACCIÓN CRÍTICA DETECTADA")
    print("El Agente desea generar y guardar un documento legal oficial.")
    print("="*50)
    
    borrador = f"""
    EN EL JUZGADO DE LO {tipo_caso.upper()}
    
    DEMANDA POR CUANTÍA DE ${monto}
    
    HECHOS:
    {argumentos}
    
    SOLICITUD:
    Se solicita al juzgado admitir la presente demanda y condenar al demandado al pago de la suma solicitada.
    
    Atentamente,
    Jusegora - Abogado
    """
    
    print("\n--- BORRADOR GENERADO POR LA IA ---")
    print(borrador)
    print("-----------------------------------")
    
    print("\n[HUMAN-IN-THE-LOOP REQUERIDO]")
    aprobacion = input("¿Aprueba usted la redacción y envío de este documento legal? (Escriba 'SI' o 'NO'): ").strip().upper()
    
    if aprobacion == "SI":
        nombre_archivo = f"demanda_{tipo_caso.lower()}_{monto}.txt"
        with open(nombre_archivo, "w", encoding="utf-8") as f:
            f.write(borrador)
        return {"status": "APROBADO", "mensaje": f"✅ EXITO: Documento guardado como '{nombre_archivo}'."}
    else:
        return {"status": "ABORTADO", "mensaje": "❌ ACCION CANCELADA POR EL HUMANO."}

herramientas = [
    {
        "type": "function",
        "function": {
            "name": "redactar_demanda_borrador",
            "description": "Usa esta herramienta cuando el usuario pida redactar, crear o enviar una demanda legal formal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tipo_caso": {"type": "string", "description": "El tipo de caso (Laboral, Civil, etc.)"},
                    "monto": {"type": "integer", "description": "El monto de la demanda"},
                    "argumentos": {"type": "string", "description": "Los hechos y argumentos legales del caso"}
                },
                "required": ["tipo_caso", "monto", "argumentos"]
            }
        }
    }
]

class AgenteSeguro:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        self.llm = Groq(api_key=api_key, timeout=15.0)
        # Le ordenamos explicitamente a la IA que SIEMPRE use la herramienta
        self.mensajes = [
            {"role": "system", "content": "Eres un abogado experto. SIEMPRE que el usuario te pida redactar una demanda, DEBES usar la herramienta 'redactar_demanda_borrador'. No generes el texto tu mismo, usa la herramienta."}
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
                
                if nombre_funcion == "redactar_demanda_borrador":
                    resultado_herramienta = redactar_demanda_borrador(
                        argumentos.get("tipo_caso"), 
                        argumentos.get("monto"), 
                        argumentos.get("argumentos")
                    )
                    
                    # --- GUARDRAIL (BARRERA DE SEGURIDAD EN PYTHON) ---
                    if resultado_herramienta["status"] == "ABORTADO":
                        print("\n" + "="*50)
                        print(f"Agente Legal: Proceso detenido por seguridad. {resultado_herramienta['mensaje']}")
                        print("El agente no puede continuar sin la autorización humana.")
                        print("="*50)
                        return 
                    
                    self.mensajes.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": nombre_funcion,
                        "content": resultado_herramienta["mensaje"]
                    })
                
            respuesta_final = self.llm.chat.completions.create(
                messages=self.mensajes,
                model="openai/gpt-oss-20b"
            )
            print(f"\nAgente Legal: {respuesta_final.choices[0].message.content}")
            
        else:
            print(f"\nAgente Legal: {mensaje_respuesta.content}")

if __name__ == "__main__":
    agente = AgenteSeguro()
    agente.preguntar("Necesito que redactes una demanda laboral por 50000 dolares. El trabajador fue despedido injustificadamente despues de 5 anos de servicio.")
