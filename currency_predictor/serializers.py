from rest_framework import serializers

from currency_predictor.models import EuroRates, PoundRates


class EuroRatesSerializer(serializers.ModelSerializer):
    class Meta:
        model = EuroRates
        fields = "__all__"


class PoundRatesSerializer(serializers.ModelSerializer):
    class Meta:
        model = PoundRates
        fields = "__all__"
