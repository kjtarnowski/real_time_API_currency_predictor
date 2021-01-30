from celery import shared_task

from currency_predictor.ML_predictor_utils import TuneReporterCallback, CurrencyPredictor
from currency_predictor.models import EuroRates, PoundRates
from real_time_API_currency_predictor.settings import CURRENCY_PREDICTOR_COMMON_DICT

currency_predictor_euro_dolar_param_dict = {
    **CURRENCY_PREDICTOR_COMMON_DICT,
    "currency_model": EuroRates,
    "callbacks": [TuneReporterCallback()],
    "get_currency_data_kwargs_dict": {"currency_position_in_web_site": 0},
    "currency_code": "EURUSD",
}

euro_predictor = CurrencyPredictor(**currency_predictor_euro_dolar_param_dict)


@shared_task(name="predict_and_get_currency_data_from_web_euro")
def predict_and_get_currency_data_from_API_euro():
    euro_predictor.predict_value_based_on_last_n_values_add_prediction_and_real_data_from_web_to_db()


@shared_task(name="fit_time_series_model_euro")
def fit_time_series_model_euro():
    euro_predictor.fit_model_based_on_n_points_data()


@shared_task(name="optimize_time_series_model_euro")
def optimize_time_series_model_euro():
    euro_predictor.optimize_model_based_on_n_points_data()


currency_predictor_pound_dollar_param_dict = {
    **CURRENCY_PREDICTOR_COMMON_DICT,
    "currency_model": PoundRates,
    "callbacks": [TuneReporterCallback()],
    "get_currency_data_kwargs_dict": {"currency_position_in_web_site": 1},
    "currency_code": "GBPUSD",
}

pound_predictor = CurrencyPredictor(**currency_predictor_pound_dollar_param_dict)


@shared_task(name="predict_and_get_currency_data_from_web_pound")
def predict_and_get_currency_data_from_API_pound():
    pound_predictor.predict_value_based_on_last_n_values_add_prediction_and_real_data_from_web_to_db()


@shared_task(name="fit_time_series_model_pound")
def fit_time_series_model_pound():
    pound_predictor.fit_model_based_on_n_points_data()


@shared_task(name="optimize_time_series_model_pound")
def optimize_time_series_model_pound():
    pound_predictor.optimize_model_based_on_n_points_data()
