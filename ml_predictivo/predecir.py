import pandas as pd
import joblib

def predecir_caso(tipo_caso: str, juez_id: str, monto: int):
    print(f"\nAnalizando caso: {tipo_caso} | Juez: {juez_id} | Monto: ${monto}")
    
    try:
        modelo = joblib.load('ml_predictivo/modelo_judicial.joblib')
        le_tipo = joblib.load('ml_predictivo/encoder_tipo.joblib')
        le_juez = joblib.load('ml_predictivo/encoder_juez.joblib')
        le_veredicto = joblib.load('ml_predictivo/encoder_veredicto.joblib')
    except FileNotFoundError:
        print("Error: Primero debes ejecutar entrenar_modelo.py")
        return

    try:
        tipo_num = le_tipo.transform([tipo_caso])[0]
        juez_num = le_juez.transform([juez_id])[0]
    except ValueError as e:
        print(f"Error: Categoria no vista en el entrenamiento. {e}")
        return

    X_nuevo = pd.DataFrame([[tipo_num, juez_num, monto]], 
                           columns=['tipo_caso_num', 'juez_num', 'monto_reclamado'])
    
    prediccion_num = modelo.predict(X_nuevo)[0]
    probabilidades = modelo.predict_proba(X_nuevo)[0]
    
    veredicto = le_veredicto.inverse_transform([prediccion_num])[0]
    prob_exito = probabilidades[1] * 100 # Probabilidad de clase 1 (Favorable)
    
    print(f"=== RESULTADO DEL ANALISIS ===")
    print(f"Veredicto Pronosticado: {veredicto}")
    print(f"Probabilidad de Exito: {prob_exito:.2f}%")

if __name__ == "__main__":
    # Escenarios de prueba
    predecir_caso("Laboral", "JUEZ-05", 30000) 
    predecir_caso("Penal", "JUEZ-12", 5000)    
    predecir_caso("Civil", "JUEZ-01", 15000)   
