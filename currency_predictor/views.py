from rest_framework import generics

from currency_predictor.models import EuroRates, PoundRates
from currency_predictor.serializers import EuroRatesSerializer, PoundRatesSerializer


class ListEuroRatesView(generics.ListAPIView):
    queryset = EuroRates.objects.all()
    serializer_class = EuroRatesSerializer


class ListPoundRatesView(generics.ListAPIView):
    queryset = PoundRates.objects.all()
    serializer_class = PoundRatesSerializer
