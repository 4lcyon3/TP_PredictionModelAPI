from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .ml_model import predecir

@csrf_exempt
def predict_view(request):
    if request.method == "POST":
        data = json.loads(request.body.decode("utf-8"))
        score = data.get("score_total")
        if score is None:
            return JsonResponse({"error": "Debe incluir 'score_total'."}, status=400)
        
        resultado = predecir(score)
        return JsonResponse(resultado)
    return JsonResponse({"message": "Use POST con JSON {'score_total': valor}."})
