from django.contrib import admin
from currency_predictor.models import EuroRates, PoundRates


@admin.register(EuroRates)
class EuroRatesAdmin(admin.ModelAdmin):
    list_display = [field.name for field in EuroRates._meta.get_fields()]


@admin.register(PoundRates)
class PoundRatesAdmin(admin.ModelAdmin):
    list_display = [field.name for field in PoundRates._meta.get_fields()]
