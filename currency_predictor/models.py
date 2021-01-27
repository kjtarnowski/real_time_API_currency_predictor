from django.db import models


class DollarRates(models.Model):
    rate = models.FloatField(blank=True, null=True)
    time = models.TimeField(blank=True, null=True)
    time_pred = models.TimeField(auto_now_add=True, blank=True, null=True)
    rate_pred = models.FloatField(blank=True, null=True)

    def __str__(self):
        return self.time_pred


class PoundRates(models.Model):
    rate = models.FloatField(blank=True, null=True)
    time = models.TimeField(blank=True, null=True)
    time_pred = models.TimeField(auto_now_add=True, blank=True, null=True)
    rate_pred = models.FloatField(blank=True, null=True)

    def __str__(self):
        return self.time_pred
