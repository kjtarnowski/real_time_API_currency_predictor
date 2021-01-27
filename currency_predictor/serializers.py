from rest_framework import serializers

from real_time_API_currency_predictor.currency_predictor.models import DollarRates, PoundRates


class EURUSDSerializer(serializers.ModelSerializer):
    class Meta:
        model = DollarRates
        fields = '__all__'


class EURGBPSerializer(serializers.ModelSerializer):
    class Meta:
        model = PoundRates
        fields = '__all__'