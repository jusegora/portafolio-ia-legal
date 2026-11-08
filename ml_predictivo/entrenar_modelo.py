import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train():
    df = pd.read_csv("ml_predictivo/casos_judiciales.csv")
    
    le_tipo = LabelEncoder()
    le_juez = LabelEncoder()
    le_verd = LabelEncoder()
    
    df['tipo_num'] = le_tipo.fit_transform(df['tipo_caso'])
    df['juez_num'] = le_juez.fit_transform(df['juez_id'])
    df['verd_num'] = le_verd.fit_transform(df['veredicto'])
    
    X = df[['tipo_num', 'juez_num', 'monto_reclamado']]
    y = df['verd_num']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    print(f"Precision: {accuracy_score(y_test, preds):.4f}")
    
    joblib.dump(clf, 'ml_predictivo/modelo.joblib')
    joblib.dump(le_tipo, 'ml_predictivo/enc_tipo.joblib')
    joblib.dump(le_juez, 'ml_predictivo/enc_juez.joblib')
    joblib.dump(le_verd, 'ml_predictivo/enc_verd.joblib')

if __name__ == "__main__":
    train()
