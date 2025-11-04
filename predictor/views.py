import pandas as pd
import numpy as np
import joblib
import os
from keras.models import load_model
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

# === Ruta base donde están los modelos ===
MODEL_DIR = os.path.join(settings.BASE_DIR, "data", "models")

# === Cargar modelos entrenados ===
def load_models():
    rf_path = os.path.join(MODEL_DIR, "rf_model.joblib")
    nn_path = os.path.join(MODEL_DIR, "nn_model.h5")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")

    if not (os.path.exists(rf_path) and os.path.exists(nn_path) and os.path.exists(scaler_path)):
        raise FileNotFoundError("⚠️ No se encontraron los modelos entrenados. Ejecuta primero 'train_all()'.")

    rf = joblib.load(rf_path)
    scaler = joblib.load(scaler_path)
    nn = load_model(nn_path)
    return rf, nn, scaler


# === Función de predicción híbrida ===
def hybrid_predict(df):
    rf, nn, scaler = load_models()

    # Asegurar que las columnas requeridas existan
    expected_cols = ["persistente", "competente", "observador"]
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(f"El CSV debe contener las columnas: {expected_cols}")

    X = df[expected_cols].values
    X_scaled = scaler.transform(X)

    # Predicciones
    rf_pred = rf.predict(X_scaled)
    nn_pred = nn.predict(X_scaled).flatten() # type: ignore

    # Predicción híbrida
    hybrid_pred = (0.6 * rf_pred + 0.4 * nn_pred)

    # Construir respuesta
    df_result = df.copy()
    df_result["rendimiento_predicho"] = hybrid_pred
    return df_result


# === Vista API ===
@csrf_exempt
def predict_view(request):
    if request.method == "POST":
        try:
            # 📁 Caso 1: CSV enviado
            if "file" in request.FILES:
                csv_file = request.FILES["file"]
                df = pd.read_csv(csv_file)

                df_result = hybrid_predict(df)
                result_json = df_result.to_dict(orient="records")
                return JsonResponse(result_json, safe=False, status=200)

            # 🧍 Caso 2: JSON individual
            elif request.body:
                import json
                body = json.loads(request.body.decode("utf-8"))

                # Convertir a DataFrame
                df = pd.DataFrame([body])
                df_result = hybrid_predict(df)
                result_json = df_result.to_dict(orient="records")[0]
                return JsonResponse(result_json, safe=False, status=200)

            else:
                return JsonResponse({"error": "No se envió ningún archivo ni JSON."}, status=400)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    else:
        return JsonResponse({"error": "Método no permitido."}, status=405)
