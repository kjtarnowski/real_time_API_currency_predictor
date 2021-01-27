from celery import shared_task

from real_time_API_currency_predictor.currency_predictor.ML_predictor_utils import TuneReporterCallback, \
    CurrencyPredictor
from real_time_API_currency_predictor.currency_predictor.models import DollarRates, PoundRates
from real_time_API_currency_predictor.real_time_API_currency_predictor.settings import CURRENCY_PREDICTOR_COMMON_DICT, \
    BASED_HYPERFILE_FILE_NAME, BASED_SCALE_FILE_NAME, BASED_MODEL_ARCH_FILE_NAME, BASED_MODEL_WEIGHTS_FILE_NAME, URL

currency_predictor_dolar_euro_param_dict = {
    **CURRENCY_PREDICTOR_COMMON_DICT,
    'url': f"{URL}EURUSD",
    'currency_rate_name': 'EURUSD',
    'currency_model': DollarRates,
    'currency_field_name': 'dolar_to_euro_ratio',
    'currency_field_name_pred': 'dolar_to_euro_ratio_pred',
    'hyperparams_file': f"EURUSD{BASED_HYPERFILE_FILE_NAME}",
    'scaler_file_name': f"EURUSD{BASED_SCALE_FILE_NAME}",
    'model_architecture_file': f"EURUSD{BASED_MODEL_ARCH_FILE_NAME}",
    'model_weight_file': f"EURUSD{BASED_MODEL_WEIGHTS_FILE_NAME}",
    'callbacks': [TuneReporterCallback()]
}

dollar_predictor = CurrencyPredictor(**currency_predictor_dolar_euro_param_dict)


@shared_task(name="predict_and_get_currency_data_from_API_dollar")
def predict_and_get_currency_data_from_API_dollar():
    dollar_predictor.predict_value_based_on_last_n_values_add_prediction_and_real_data_from_API_to_db()


@shared_task(name="fit_time_series_model_dollar")
def fit_time_series_model_dollar():
    dollar_predictor.fit_model_based_on_n_points_data()


@shared_task(name="optimize_time_series_model_dollar")
def optimize_time_series_model_dollar():
    dollar_predictor.optimize_model_based_on_n_points_data()


currency_predictor_pound_euro_param_dict = {
    **CURRENCY_PREDICTOR_COMMON_DICT,
    'url': f"{URL}EURGBP",
    'currency_rate_name': 'EURGBP',
    'currency_model': PoundRates,
    'hyperparams_file': f"EURGBP{BASED_HYPERFILE_FILE_NAME}",
    'scaler_file_name': f"EURGBP{BASED_SCALE_FILE_NAME}",
    'model_architecture_file': f"EURGBP{BASED_MODEL_ARCH_FILE_NAME}",
    'model_weight_file': f"EURGBP{BASED_MODEL_WEIGHTS_FILE_NAME}",
    'callbacks': [TuneReporterCallback()]
}

pound_predictor = CurrencyPredictor(**currency_predictor_pound_euro_param_dict)


@shared_task(name="predict_and_get_currency_data_from_API_pound")
def predict_and_get_currency_data_from_API_pound():
    pound_predictor.predict_value_based_on_last_n_values_add_prediction_and_real_data_from_API_to_db()


@shared_task(name="fit_time_series_model_pound")
def fit_time_series_model_pound():
    pound_predictor.fit_model_based_on_n_points_data()


@shared_task(name="optimize_time_series_model_pound")
def optimize_time_series_model_pound():
    pound_predictor.optimize_model_based_on_n_points_data()
