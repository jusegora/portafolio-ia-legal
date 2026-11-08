import pandas as pd
import joblib

def predecir(tipo, juez, monto):
    clf = joblib.load('ml_predictivo/modelo.joblib')
    le_tipo = joblib.load('ml_predictivo/enc_tipo.joblib')
    le_juez = joblib.load('ml_predictivo/enc_juez.joblib')
    le_verd = joblib.load('ml_predictivo/enc_verd.joblib')

    t_num = le_tipo.transform([tipo])[0]
    j_num = le_juez.transform([juez])[0]

    X = pd.DataFrame([[t_num, j_num, monto]], columns=['tipo_num', 'juez_num', 'monto_reclamado'])
    
    pred = clf.predict(X)[0]
    prob = clf.predict_proba(X)[0][1] * 100
    
    print(f"Caso: {tipo} | Juez: {juez} | Monto: {monto}")
    print(f"Resultado: {le_verd.inverse_transform([pred])[0]} ({prob:.2f}% exito)")

if __name__ == "__main__":
    predecir("Laboral", "JUEZ-05", 30000)
    predecir("Penal", "JUEZ-12", 5000)
