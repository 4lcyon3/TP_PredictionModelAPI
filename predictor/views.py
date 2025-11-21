from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import pandas as pd
import joblib
import os
from keras.models import load_model
from django.conf import settings

MODEL_DIR = os.path.join(settings.BASE_DIR, "data", "models")


def load_models():
    rf = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    nn = load_model(os.path.join(MODEL_DIR, "nn_model.h5"), compile=False)
    nn.compile(optimizer="adam", loss="mse", metrics=["mae"]) # type: ignore
    return rf, nn, scaler


@csrf_exempt
def predict_view(request):
    if request.method != "POST":
        return JsonResponse({"error": "Método no permitido"}, status=405)

    try:
        data = json.loads(request.body)

        df = pd.DataFrame([data])
        rf, nn, scaler = load_models()

        X = scaler.transform(df.values)

        rf_pred = rf.predict(X)
        nn_pred = nn.predict(X).flatten() # type: ignore
        hybrid_pred = (0.6 * rf_pred + 0.4 * nn_pred)[0]

        return JsonResponse({"prediction": float(hybrid_pred)})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
