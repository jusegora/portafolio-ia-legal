import pandas as pd
import numpy as np
import os

def generar_dataset_sintetico(n_registros: int = 5000, ruta_salida: str = "ml_predictivo/casos_judiciales.csv"):
    print(f"Generando {n_registros} casos judiciales sinteticos...")
    np.random.seed(42)
    
    tipos_caso = np.random.choice(['Laboral', 'Civil', 'Penal', 'Administrativo'], n_registros, p=[0.4, 0.3, 0.2, 0.1])
    jueces = np.random.choice([f"JUEZ-{i:02d}" for i in range(1, 21)], n_registros)
    montos = np.random.randint(1000, 50000, n_registros)
    
    probabilidad_base = np.where(tipos_caso == 'Laboral', 0.65, 0.45)
    probabilidad_base = np.where((tipos_caso == 'Laboral') & (montos > 25000), 0.80, probabilidad_base)
    probabilidad_base = np.where(tipos_caso == 'Penal', 0.30, probabilidad_base)
    
    resultados = np.random.binomial(1, probabilidad_base)
    veredictos = np.where(resultados == 1, 'Favorable', 'Desfavorable')
    
    df = pd.DataFrame({
        'tipo_caso': tipos_caso,
        'juez_id': jueces,
        'monto_reclamado': montos,
        'veredicto': veredictos
    })
    
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    df.to_csv(ruta_salida, index=False)
    print(f"Dataset guardado exitosamente en: {ruta_salida}")
    print("\nPrimeros 5 registros:")
    print(df.head())

if __name__ == "__main__":
    generar_dataset_sintetico()
