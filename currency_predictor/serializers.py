from rest_framework import serializers

from real_time_API_currency_predictor.currency_predictor.models import EURUSD, EURGBP


class EURUSDSerializer(serializers.ModelSerializer):
    class Meta:
        model = EURUSD
        fields = '__all__'


class EURGBPSerializer(serializers.ModelSerializer):
    class Meta:
        model = EURGBP
        fields = '__all__'