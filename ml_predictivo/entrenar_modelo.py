import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def entrenar_modelo(ruta_datos: str = "ml_predictivo/casos_judiciales.csv"):
    print("Cargando datos...")
    df = pd.read_csv(ruta_datos)
    
    print("Preprocesando variables categoricas...")
    le_tipo = LabelEncoder()
    le_juez = LabelEncoder()
    le_veredicto = LabelEncoder()
    
    df['tipo_caso_num'] = le_tipo.fit_transform(df['tipo_caso'])
    df['juez_num'] = le_juez.fit_transform(df['juez_id'])
    df['veredicto_num'] = le_veredicto.fit_transform(df['veredicto'])
    
    X = df[['tipo_caso_num', 'juez_num', 'monto_reclamado']]
    y = df['veredicto_num']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Entrenando modelo Random Forest...")
    modelo = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    modelo.fit(X_train, y_train)
    
    predicciones = modelo.predict(X_test)
    precision = accuracy_score(y_test, predicciones)
    
    print(f"\nPrecision del modelo (Accuracy): {precision:.4f}")
    print("\nReporte de Clasificacion:")
    print(classification_report(y_test, predicciones, target_names=le_veredicto.classes_))
    
    joblib.dump(modelo, 'ml_predictivo/modelo_judicial.joblib')
    joblib.dump(le_tipo, 'ml_predictivo/encoder_tipo.joblib')
    joblib.dump(le_juez, 'ml_predictivo/encoder_juez.joblib')
    joblib.dump(le_veredicto, 'ml_predictivo/encoder_veredicto.joblib')
    
    print("\nModelo y codificadores guardados exitosamente.")

if __name__ == "__main__":
    entrenar_modelo()
