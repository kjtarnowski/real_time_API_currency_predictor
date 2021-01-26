from rest_framework import generics

from real_time_API_currency_predictor.currency_predictor.models import EURUSD, EURGBP
from real_time_API_currency_predictor.currency_predictor.serializers import EURUSDSerializer, EURGBPSerializer


class ListCurrencyEURUSDView(generics.ListAPIView):
    queryset = EURUSD.objects.all()
    serializer_class = EURUSDSerializer


class ListCurrencyEURGBPView(generics.ListAPIView):
    queryset = EURGBP.objects.all()
    serializer_class = EURGBPSerializer
