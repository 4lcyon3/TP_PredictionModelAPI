import pandas as pd
import numpy as np
import json

def preparar_dataset(ruta_csv: str, salida_csv: str = "dataset_entrenamiento_final.csv"):
    df = pd.read_csv(ruta_csv, sep=";", encoding="latin1")

    # Verificar columna principal
    if "PROMEDIO_X_SECCION" not in df.columns:
        raise ValueError("No se encontró la columna 'PROMEDIO_X_SECCION'.")

    # Buscar columna de nombre del alumno
    name_cols = [col for col in df.columns if any(k in col.lower() for k in ["nombre", "alumno", "estudiante", "institu"])]
    if name_cols:
        col_name = name_cols[0]
    else:
        df["nombre_generado"] = [f"Alumno_{i+1}" for i in range(len(df))]
        col_name = "nombre_generado"

    df_final = pd.DataFrame()
    df_final["student_name"] = df[col_name].astype(str)
    df_final["score_total"] = pd.to_numeric(df["PROMEDIO_X_SECCION"], errors="coerce")

    # Historial (4 repeticiones)
    df_final["history"] = df_final["score_total"].apply(lambda x: json.dumps([round(x, 2)] * 4))

    # Clasificación de rendimiento
    mean_score = df_final["score_total"].mean()
    df_final["final_performance"] = np.where(
        df_final["score_total"] >= mean_score,
        "aprobará",
        "no_aprobará"
    )

    # Guardar dataset limpio
    df_final.to_csv(salida_csv, index=False, encoding="utf-8-sig")
    print(f"Dataset preparado y guardado en {salida_csv}")
    return df_final
