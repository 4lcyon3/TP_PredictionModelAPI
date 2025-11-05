import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw" / "3ro.csv"
PROCESSED = BASE / "data" / "processed"
MODELS = BASE / "data" / "models"

PROCESSED.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)


def preparar_dataset(path_csv=None):
    path_csv = Path(path_csv) if path_csv else RAW

    df_raw = pd.read_csv(path_csv, sep=";", encoding="latin1", header=None)

    header_row = df_raw[df_raw.iloc[:, 0].astype(str).str.contains("N", na=False)].index[0]
    df_header = df_raw.iloc[header_row]
    df_data = df_raw.iloc[header_row + 2:] 

    capacidades_row = df_raw.iloc[header_row + 1].fillna("")

    df_data.columns = df_header
    df_data = df_data.reset_index(drop=True)

    p_cols = [c for c in df_data.columns if re.match(r"^P\s*\d+", str(c).strip(), re.I)]

    if not p_cols:
        raise ValueError("No se encontraron columnas válidas (P <número>) en el CSV.")

    df_data[p_cols] = df_data[p_cols].apply(pd.to_numeric, errors="coerce")

    map_c = {c: capacidades_row[i] for i, c in enumerate(df_data.columns) if c in p_cols}

    c1_cols = [c for c, t in map_c.items() if str(t).strip().upper() == "C1"]
    c2_cols = [c for c, t in map_c.items() if str(t).strip().upper() == "C2"]
    c3_cols = [c for c, t in map_c.items() if str(t).strip().upper() == "C3"]

    df_out = pd.DataFrame()
    df_out["student_name"] = [f"alumno{i+1}" for i in range(len(df_data))]
    df_out["persistente"] = df_data[c1_cols].sum(axis=1)
    df_out["competente"] = df_data[c2_cols].sum(axis=1)
    df_out["observador"] = df_data[c3_cols].sum(axis=1)
    df_out["score_total"] = df_out[["persistente", "competente", "observador"]].sum(axis=1)
    df_out["history"] = df_out["score_total"].apply(lambda x: json.dumps([round(float(x), 2)] * 4))
    df_out["final_performance"] = ""

    out_path = PROCESSED / "dataset_limpio.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    scaler = StandardScaler()
    scaler.fit(df_out[["persistente", "competente", "observador"]])
    joblib.dump(scaler, MODELS / "scaler.joblib")
    return df_out


def procesar_y_guardar_dataset():
    preparar_dataset(RAW)


if __name__ == "__main__":
    procesar_y_guardar_dataset()
