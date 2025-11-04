import pandas as pd
import numpy as np
import json
import re
import sys
from pathlib import Path

def find_row_with_text(df_raw, text):
    """Busca índice de la primera fila que contenga `text` en alguna celda."""
    for i, row in df_raw.iterrows():
        for v in row:
            if isinstance(v, str) and text.lower() in v.lower():
                return i
    return None

def parse_c_header_row(row):
    """Devuelve lista de tokens de la fila (limpios)."""
    tokens = []
    for v in row:
        if pd.isna(v):
            tokens.append("")
        else:
            tokens.append(str(v).strip())
    return tokens

def extract_numeric(cell):
    if pd.isna(cell):
        return np.nan
    s = str(cell).strip()
    s = s.replace(",", ".")
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group(0)) if m else np.nan

def clean_4ro_csv(path_in="4to.csv", path_out="dataset_4to_clean.csv"):
    p = Path(path_in)
    if not p.exists():
        raise FileNotFoundError(f"No se encuentra {path_in}")

    df_raw = pd.read_csv(p, sep=';', encoding='latin1', header=None, dtype=str, low_memory=False)

    idx_cap = find_row_with_text(df_raw, "CAPACIDADES")
    if idx_cap is None:
        idx_cap = find_row_with_text(df_raw, "CAPACIDAD")
    if idx_cap is None:
        raise ValueError("No se encontró la fila 'CAPACIDADES' en el CSV. Verifica la estructura del archivo.")
    header_row_idx = idx_cap + 1
    subheader_row_idx = idx_cap + 2

    header_candidate_idx = None
    for test_i in range(0, min(10, len(df_raw))):
        row_text = " ".join([str(x) for x in df_raw.loc[test_i].tolist() if not pd.isna(x)])
        if "N°" in row_text or "INSTITUCION" in row_text.upper() or "RED" in row_text.upper():
            header_candidate_idx = test_i
            break
    if header_candidate_idx is None:
        header_candidate_idx = 0

    df = pd.read_csv(p, sep=';', encoding='latin1', header=header_candidate_idx, low_memory=False)

    df.columns = [str(c).strip() for c in df.columns]

    subrow = None
    if subheader_row_idx < len(df_raw):
        subrow = parse_c_header_row(df_raw.loc[subheader_row_idx])
    else:
        for i in range(0, min(15, len(df_raw))):
            tokens = parse_c_header_row(df_raw.loc[i])
            cnt = sum(1 for t in tokens if re.match(r"^C\d+$", t.strip().upper()))
            if cnt >= 3:
                subrow = tokens
                break
    col_to_C = {}
    if subrow:
        
        raw_header_row = parse_c_header_row(df_raw.loc[header_candidate_idx])
        col_index_map = {}
        for idx, col in enumerate(raw_header_row):
            col_index_map[idx] = idx
        for i, colname in enumerate(df.columns):
            found_idx = None
            for j, rawcol in enumerate(raw_header_row):
                if rawcol and str(colname).strip().lower().startswith(str(rawcol).strip().lower()):
                    found_idx = j
                    break
            if found_idx is None:
                found_idx = i if i < len(subrow) else None
            if found_idx is not None and found_idx < len(subrow):
                token = subrow[found_idx].strip()
                if re.match(r"^C\d+$", token.upper()):
                    col_to_C[colname] = token.upper()
    if not col_to_C:
        p_cols = [c for c in df.columns if re.match(r"^P\s*\d+", str(c).strip(), re.I) or re.match(r"^P\d+", str(c).strip(), re.I)]
        if subrow and len(subrow) == len(df.columns):
            for idx, col in enumerate(df.columns):
                token = subrow[idx].strip()
                if token.upper() in ("C1","C2","C3"):
                    col_to_C[col] = token.upper()
        else:
            for i in range(0, min(20, len(df_raw))):
                tokens = parse_c_header_row(df_raw.loc[i])
                cntC = sum(1 for t in tokens if re.match(r"^C\d+$", str(t).strip().upper()))
                if cntC >= 3:
                    for j, t in enumerate(tokens):
                        if re.match(r"^C\d+$", str(t).strip().upper()):
                            if j < len(df.columns):
                                col_to_C[df.columns[j]] = t.strip().upper()
                    break

    name_candidates = [c for c in df.columns if any(k in c.lower() for k in ["instit", "ie", "institucion", "nombre", "estudiante"])]
    if name_candidates:
        name_col = name_candidates[0]
    else:
        name_col = df.columns[0]

    out_rows = []
    alumno_idx = 0
    for _, row in df.iterrows():
        caps = {"C1":0.0, "C2":0.0, "C3":0.0}
        for col, ccode in col_to_C.items():
            if col in df.columns:
                val = extract_numeric(row[col])
                if not np.isnan(val):
                    caps[ccode] += val
        if not col_to_C:
            pcols = [c for c in df.columns if re.match(r"^P\s*\d+|^P\d+", str(c), re.I)]
            if pcols:
                vals = [extract_numeric(row[c]) for c in pcols]
                vals = [0 if np.isnan(v) else v for v in vals]
                total = sum(vals)
                n = len(vals)
                for i, v in enumerate(vals):
                    if i < n/3:
                        caps["C1"] += v
                    elif i < 2*n/3:
                        caps["C2"] += v
                    else:
                        caps["C3"] += v

        score_total = None
        if "PROMEDIO_X_SECCION" in df.columns:
            try:
                score_total = float(str(row["PROMEDIO_X_SECCION"]).replace("%","").replace(",",".")) 
            except Exception:
                score_total = np.nan
        if score_total is None or (isinstance(score_total, float) and np.isnan(score_total)):
            score_total = caps["C1"] + caps["C2"] + caps["C3"]

        alumno_idx += 1
        student_name = f"alumno{alumno_idx}"
        history = [round(float(score_total),2)]*4
        out_rows.append({
            "student_name": student_name,
            "persistente": round(float(caps["C1"]),2),
            "competente": round(float(caps["C2"]),2),
            "observador": round(float(caps["C3"]),2),
            "score_total": float(score_total),
            "history": json.dumps(history),
            "final_performance": ""
        })

    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(path_out, index=False, encoding='utf-8-sig')
    print("Archivo generado:", path_out)
    return df_out

if __name__ == "__main__":
    in_csv = "4to.csv"
    out_csv = "dataset_4to_clean.csv"
    print("Procesando", in_csv)
    df_out = clean_4ro_csv(in_csv, out_csv)
    print(df_out.head())
