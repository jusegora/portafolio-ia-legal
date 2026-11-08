import pandas as pd
import numpy as np
import os

def crear_dataset(n=5000):
    np.random.seed(42)
    
    tipos = np.random.choice(['Laboral', 'Civil', 'Penal', 'Administrativo'], n, p=[0.4, 0.3, 0.2, 0.1])
    jueces = np.random.choice([f"JUEZ-{i:02d}" for i in range(1, 21)], n)
    montos = np.random.randint(1000, 50000, n)
    
    # Probabilidad base
    prob = np.where(tipos == 'Laboral', 0.65, 0.45)
    prob = np.where((tipos == 'Laboral') & (montos > 25000), 0.80, prob)
    prob = np.where(tipos == 'Penal', 0.30, prob)
    
    resultados = np.random.binomial(1, prob)
    veredictos = np.where(resultados == 1, 'Favorable', 'Desfavorable')
    
    df = pd.DataFrame({
        'tipo_caso': tipos,
        'juez_id': jueces,
        'monto_reclamado': montos,
        'veredicto': veredictos
    })
    
    os.makedirs('ml_predictivo', exist_ok=True)
    df.to_csv('ml_predictivo/casos_judiciales.csv', index=False)
    print("Dataset creado.")

if __name__ == "__main__":
    crear_dataset()
