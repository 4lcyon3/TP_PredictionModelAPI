# predictor/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .ml_model import predict

@csrf_exempt
def predict_view(request):
    if request.method != "POST":
        return JsonResponse({"detail":"Use POST"}, status=405)
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error":"Invalid JSON"}, status=400)
    try:
        res = predict(payload)
        return JsonResponse(res)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
