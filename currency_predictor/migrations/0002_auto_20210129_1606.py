# Generated by Django 3.1.4 on 2021-01-29 16:06

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('currency_predictor', '0001_initial'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='DollarRates',
            new_name='EuroRates',
        ),
    ]
