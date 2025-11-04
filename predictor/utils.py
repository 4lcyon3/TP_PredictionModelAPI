# predictor/utils.py
import joblib
import json
from pathlib import Path
import numpy as np
import keras

BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "data" / "models"

def load_artifacts():
    """
    Carga rf_model, scaler, nn_model y lstm_model desde data/models
    """
    rf = None
    scaler = None
    nn = None
    lstm = None
    try:
        rf = joblib.load(MODELS_DIR / "rf_model.joblib")
    except Exception:
        rf = None
    try:
        scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    except Exception:
        scaler = None
    try:
        nn = keras.models.load_model(str(MODELS_DIR / "nn_model.h5"))
    except Exception:
        nn = None
    try:
        lstm = keras.models.load_model(str(MODELS_DIR / "lstm_model.h5"))
    except Exception:
        lstm = None
    return rf, scaler, nn, lstm

def pad_or_truncate_history(hist, timesteps=4):
    """
    hist: list of scalars or list of [p,c,o] lists
    Returns list of length timesteps
    """
    if hist is None:
        return [0.0]*timesteps
    if isinstance(hist, str):
        try:
            hist = json.loads(hist)
        except Exception:
            hist = [float(x) for x in hist.split(",") if x.strip()!='']
    if not isinstance(hist, (list, tuple)):
        return [float(hist)] * timesteps
    if len(hist) >= timesteps:
        return hist[-timesteps:]
    # pad with last value
    pad = [hist[-1]]*(timesteps - len(hist))
    return pad + hist # type: ignore
