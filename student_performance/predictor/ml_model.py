import os
import joblib
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Cargar artefactos una sola vez al iniciar Django
rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.joblib"))
tf_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "tf_model"))
preproc = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))

def scores_to_feature_vector(scores):
    d = {}
    for s in scores:
        g = s['group_type']
        d[g] = d.get(g, 0.0) + float(s['score'])

    feature_cols = preproc['feature_cols']
    vec = np.array([d.get(col, 0.0) for col in feature_cols]).reshape(1,-1)
    scaled = preproc['scaler'].transform(vec)
    return scaled

def predict_student(scores):
    X = scores_to_feature_vector(scores)
    pred_rf = rf_model.predict(X)[0]
    pred_tf = tf_model.predict(X).reshape(-1)[0]
    pred_ens = float((pred_rf + pred_tf) / 2.0)
    return {
        "pred_rf": float(pred_rf),
        "pred_tf": float(pred_tf),
        "pred_ensemble": pred_ens
    }
