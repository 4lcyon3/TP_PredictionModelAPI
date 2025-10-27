from rest_framework import serializers

class ScoreSerializer(serializers.Serializer):
    group_type = serializers.CharField()
    score = serializers.FloatField()

class PredictRequestSerializer(serializers.Serializer):
    student_name = serializers.CharField(required=False)
    scores = ScoreSerializer(many=True)
