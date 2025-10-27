from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import PredictRequestSerializer
from .ml_model import predict_student

class PredictView(APIView):
    def post(self, request):
        serializer = PredictRequestSerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            prediction = predict_student(data["scores"])
            return Response({
                "student_name": data.get("student_name", "unknown"),
                **prediction
            })
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
