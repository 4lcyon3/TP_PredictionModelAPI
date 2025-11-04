import keras
import os, joblib, json, numpy as np
from pathlib import Path
import tensorflow as tf

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models_hybrid"

_rf = None
_scaler = None
_tf = None
_le = None

def load_models(model_dir=None):
    global _rf, _scaler, _tf, _le
    md = Path(model_dir) if model_dir else MODEL_DIR
    if (md / "random_forest.joblib").exists():
        _rf = joblib.load(md / "random_forest.joblib")
    if (md / "scaler.joblib").exists():
        _scaler = joblib.load(md / "scaler.joblib")
    if (md / "tf_model.keras").exists():
        _tf = keras.models.load_model(str(md / "tf_model.keras"))
    if (md / "label_encoder.joblib").exists():
        _le = joblib.load(md / "label_encoder.joblib")

def features_from_payload(payload):
    """
    payload can contain:
      - persistente, competente, observador (preferred)
      - or score_total and history
    returns X vector (1,n_features) and extras
    """
    p = payload.get("persistente", None)
    c = payload.get("competente", None)
    o = payload.get("observador", None)
    score_total = payload.get("score_total", None)

    hist = payload.get("history", [])
    if isinstance(hist, str):
        try:
            hist = json.loads(hist)
        except Exception:
            hist = [float(x) for x in hist.split(",") if x.strip()!='']

    if p is None or c is None or o is None:
        if score_total is None and hist:
            score_total = float(hist[-1])
        p = float(p) if p is not None else 0.0
        c = float(c) if c is not None else 0.0
        o = float(o) if o is not None else 0.0
    X = np.array([[float(p), float(c), float(o), float(score_total if score_total is not None else (p + c + o))]])
    # scale
    if _scaler is not None:
        Xs = _scaler.transform(X)
    else:
        Xs = X
    extras = {"persistente": p, "competente": c, "observador": o, "score_total": score_total, "history": hist}
    return Xs, extras

def predict(payload, model_dir=None):
    if _rf is None:
        load_models(model_dir=model_dir)
    if _rf is None:
        raise RuntimeError("Model artifacts not loaded. Run training and place artifacts in predictor/models_hybrid")
    Xs, extras = features_from_payload(payload)
    pred_rf_prob = float(_rf.predict_proba(Xs)[0,1])
    pred_tf_prob = float(_tf.predict(Xs).reshape(-1)[0]) if _tf is not None else pred_rf_prob
    prob = (pred_rf_prob + pred_tf_prob) / 2.0
    label = None
    if _le is not None:
        label_idx = int(prob >= 0.5)
        try:
            label = _le.inverse_transform([label_idx])[0]
        except Exception:
            label = "aprobará" if label_idx==1 else "no_aprobará"
    else:
        label = "aprobará" if prob >= 0.5 else "no_aprobará"
    out = {
        "student_name": payload.get("student_name"),
        "probability": prob,
        "label": label,
        "pred_rf": pred_rf_prob,
        "pred_tf": pred_tf_prob,
        "features": extras
    }
    return out
