import os
import shutil
import tempfile
import json
from datetime import datetime
from typing import Callable, Dict, List, Tuple
from dataclasses import dataclass
from urllib.request import urlopen, Request


from bs4 import BeautifulSoup
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest import ConcurrencyLimiter

from django.db import models


def webscrap_currency_data_bid_and_time_from_investing_com_bid(currency_position_in_web_site):
    req = Request("https://www.investing.com/currencies/single-currency-crosses", headers={"User-Agent": "Mozilla/5.0"})
    html = urlopen(req).read()
    bs = BeautifulSoup(html, "html.parser")
    currency = bs.find("tbody").find_all("tr")[currency_position_in_web_site]
    bid = currency.find("td", class_=f"pid-{currency_position_in_web_site+1}-bid").text
    time = datetime.utcnow().replace(microsecond=0).time()
    return bid, time


def scale_and_reshape_data(data, scaler=None):
    if not scaler:
        scaler = MinMaxScaler()
    data_reshaped = data.to_numpy().reshape(-1, 1)
    data_scaled = scaler.fit_transform(data_reshaped)
    return data_scaled, scaler


def preprocess_time_series_data(data_transformed, n_steps):
    hist = []
    target = []
    data = data_transformed.tolist()

    for i in range(len(data) - n_steps):
        x = data[i : i + n_steps]
        y = data[i + n_steps]
        hist.append(x)
        target.append(y)

    hist = np.array(hist).reshape((len(hist), n_steps, 1))
    target = np.array(target).reshape(-1, 1)
    return hist, target


def preprocess_time_series_data_one_sample_for_prediction(data, n_steps, sc):
    hist = np.array(data).reshape(-1, 1)
    hist_scaled = sc.transform(hist)
    hist_scaled = hist_scaled.reshape((1, n_steps, 1))
    return hist_scaled


def create_dataset(data_train, data_test, n_steps):
    scaled_data_train, train_scaler = scale_and_reshape_data(data_train)
    scaled_data_test, _ = scale_and_reshape_data(data_test, scaler=train_scaler)

    X_train, y_train = preprocess_time_series_data(scaled_data_train, n_steps)
    X_test, y_test = preprocess_time_series_data(scaled_data_test, n_steps)

    return X_train, y_train, X_test, y_test, train_scaler


def create_model(gru_units, rec_dropout, n_steps=10):
    model_gru = tf.keras.Sequential()
    model_gru.add(layers.GRU(units=gru_units, input_shape=(n_steps, 1), recurrent_dropout=rec_dropout))
    model_gru.add(layers.Dense(units=1))

    model_gru.compile(optimizer="adam", loss="mean_squared_error")

    return model_gru


class TuneReporterCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.iteration = 0
        super().__init__()

    def on_epoch_end(self, batch, logs={}):
        self.iteration += 1
        tune.report(
            keras_info=logs,
            val_accuracy=logs.get("val_accuracy"),
            val_loss=logs.get("val_loss"),
            loss=logs.get("loss"),
        )


def train_model(
    config,
    data_train=None,
    data_test=None,
    create_model=None,
    model_kwargs_str=None,
    callbacks=None,
    epochs=None,
    n_steps=None,
):
    X_train, y_train, X_test, y_test, _ = create_dataset(data_train, data_test, n_steps)

    model_kwargs = eval(model_kwargs_str)
    model = create_model(n_steps=n_steps, **model_kwargs)

    model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_test, y_test),
        verbose=1,
        batch_size=config["batch_size"],
        epochs=epochs,
        callbacks=callbacks,
    )


def optimize_hyperparameters(
    train_model,
    create_model,
    data_train,
    data_test,
    search_space,
    model_kwargs_str,
    callbacks,
    hyperparams_file_name,
    random_seed,
    model_path,
    epochs,
    n_steps,
    num_samples_optim,
):
    tmp_dir = tempfile.TemporaryDirectory(dir=os.getcwd())

    ray.shutdown()
    ray.init(log_to_driver=False, local_mode=True)

    search_alg = HyperOptSearch(random_state_seed=random_seed)
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=1)
    scheduler = AsyncHyperBandScheduler(time_attr="training_iteration", grace_period=10)

    analysis = tune.run(
        tune.with_parameters(
            train_model,
            data_train=data_train,
            data_test=data_test,
            create_model=create_model,
            model_kwargs_str=model_kwargs_str,
            callbacks=callbacks,
            epochs=epochs,
            n_steps=n_steps,
        ),
        verbose=1,
        config=search_space,
        search_alg=search_alg,
        scheduler=scheduler,
        resources_per_trial={"cpu": os.cpu_count(), "gpu": 0},
        metric="val_loss",
        mode="min",
        name="ray_tune_keras_hyperopt_gru",
        local_dir=tmp_dir.name,
        num_samples=num_samples_optim,
    )

    shutil.rmtree(tmp_dir)

    best_params = analysis.get_best_config(metric="val_loss", mode="min")
    with open(os.path.join(model_path, hyperparams_file_name), "w") as f:
        json.dump(best_params, f)


@dataclass
class CurrencyPredictor:
    currency_model: models.Model
    currency_field_name: str
    currency_field_name_pred: str
    n_points_model: int
    n_steps: int
    n_points_training: int
    callbacks: list
    train_model: Callable
    create_model: Callable
    optimize_hyperparameters: Callable
    search_space: Dict[str, str]
    model_kwargs_str: str
    training_paramters_list: List[str]
    epochs: int
    random_seed: int
    num_samples_optim: int
    model_path: str
    pred_fit_optim_time_offset_tuple: Tuple[int, int, int]
    prediction_step: int
    get_currency_data: Callable
    get_currency_data_kwargs_dict: dict
    currency_code: str

    def get_last_n_points_data_from_db(self, n_points):
        last_ten_qs = self.currency_model.objects.all().order_by("-id")[:(n_points)]
        data_dict = {
            self.currency_field_name: last_ten_qs.values_list(self.currency_field_name, flat=True)[::-1],
            "time": last_ten_qs.values_list("time", flat=True)[::-1],
        }
        table_euro_usd = pd.DataFrame(data_dict)
        table_euro_usd["time_dt"] = pd.to_datetime(table_euro_usd["time"], format="%H:%M:%S").dt.time
        data = table_euro_usd[self.currency_field_name]
        return data

    def load_hyperparams_from_file(self):
        with open(os.path.join(self.model_path, f"{self.currency_code}_current_optimized_hyerparams.json"), "r") as f:
            best_params = json.load(f)
        return best_params

    def separate_training_params_from_model_params(self, params):
        training_params = {}
        model_params = {}
        for k, v in params.items():
            if k in self.training_paramters_list:
                training_params[k] = v
            else:
                model_params[k] = v
        return model_params, training_params

    def save_scaler_model_architecture_and_weight(self, model, scaler):
        joblib.dump(scaler, os.path.join(self.model_path, f"{self.currency_code}_current_scaler.pkl"))
        model_json = model.to_json()
        with open(
            os.path.join(self.model_path, f"{self.currency_code}_current_model_architecture.json"), "w"
        ) as json_file:
            json_file.write(model_json)
        model.save_weights(os.path.join(self.model_path, f"{self.currency_code}_current_model_weights.h5"))

    def load_scaler_model_architecture_and_weight(self):
        scaler = joblib.load(os.path.join(self.model_path, f"{self.currency_code}_current_scaler.pkl"))

        with open(
            os.path.join(self.model_path, f"{self.currency_code}_current_model_architecture.json"), "r"
        ) as json_file:
            loaded_model_json = json_file.read()

        loaded_model = keras.models.model_from_json(loaded_model_json)
        loaded_model.load_weights(os.path.join(self.model_path, f"{self.currency_code}_current_model_weights.h5"))

        return scaler, loaded_model

    def split_data_to_train_and_test(self, data, spilt_point):
        data_train = data[:spilt_point]
        data_test = data[spilt_point:]
        return data_train, data_test

    def is_enough_data_point_to_start_process(self, offset):
        n_points = self.currency_model.objects.all().count()
        return n_points > (self.n_points_model + offset)

    def load_model_and_scaler_scale_predict_data_inverse_scaling(self, data):
        scaler, loaded_model = self.load_scaler_model_architecture_and_weight()
        hist_scaled = preprocess_time_series_data_one_sample_for_prediction(data, self.n_steps, scaler)
        prediction_one_sample = loaded_model.predict(hist_scaled)
        prediction_one_sample_in_scale = scaler.inverse_transform(prediction_one_sample)
        return prediction_one_sample_in_scale

    def save_model_object_attr_in_db_from_list_of_tuples_name_attr(self, obj, list_of_tuples):
        for tup in list_of_tuples:
            setattr(obj, tup[0], tup[1])
        obj.save()

    def get_currency_data_from_web_save_in_db(self, offset):
        obj = self.currency_model.objects.all().order_by("-id")[offset]
        currency_rate, time_getting_data = self.get_currency_data(**self.get_currency_data_kwargs_dict)

        self.save_model_object_attr_in_db_from_list_of_tuples_name_attr(
            obj, [("time", time_getting_data), (self.currency_field_name, currency_rate)]
        )

    def create_empty_currency_item_in_db_for_prediction_and_get_web_data_from_previous_point(self):
        self.currency_model.objects.create()
        n_points = self.currency_model.objects.all().count()
        if n_points > 1:
            self.get_currency_data_from_web_save_in_db(self.prediction_step)

    def predict_value_based_on_last_n_values_add_prediction_and_real_data_from_web_to_db(self):
        self.create_empty_currency_item_in_db_for_prediction_and_get_web_data_from_previous_point()
        if self.is_enough_data_point_to_start_process(self.pred_fit_optim_time_offset_tuple[0]):
            data = self.get_last_n_points_data_from_db(self.n_steps + self.prediction_step)[: self.n_steps]
            prediction_one_sample_in_scale = self.load_model_and_scaler_scale_predict_data_inverse_scaling(data)
        else:
            prediction_one_sample_in_scale = 0

        obj = self.currency_model.objects.all().last()

        self.save_model_object_attr_in_db_from_list_of_tuples_name_attr(
            obj, [(self.currency_field_name_pred, prediction_one_sample_in_scale)]
        )
        print(data)

    def fit_model_based_on_n_points_data(self):
        if self.is_enough_data_point_to_start_process(self.pred_fit_optim_time_offset_tuple[1]):
            data = self.get_last_n_points_data_from_db(self.n_points_model + self.prediction_step)[
                : self.n_points_model
            ]
            data_train, data_test = self.split_data_to_train_and_test(data, self.n_points_training)

            X_train, y_train, X_test, y_test, scaler = create_dataset(data_train, data_test, self.n_steps)

            best_params = self.load_hyperparams_from_file()
            model_params_dict, training_params_dict = self.separate_training_params_from_model_params(best_params)

            model = self.create_model(**model_params_dict)
            model.fit(
                x=X_train,
                y=y_train,
                validation_data=(X_test, y_test),
                verbose=1,
                epochs=self.epochs,
                **training_params_dict,
            )

            self.save_scaler_model_architecture_and_weight(model, scaler)

    def optimize_model_based_on_n_points_data(self):
        if self.is_enough_data_point_to_start_process(self.pred_fit_optim_time_offset_tuple[2]):
            data = self.get_last_n_points_data_from_db(self.n_points_model + self.prediction_step)[
                : self.n_points_model
            ]
            data_train, data_test = self.split_data_to_train_and_test(data, self.n_points_training)

            hyperparams_file = f"{self.currency_code}_current_optimized_hyerparams.json"

            self.optimize_hyperparameters(
                self.train_model,
                self.create_model,
                data_train,
                data_test,
                self.search_space,
                self.model_kwargs_str,
                self.callbacks,
                hyperparams_file,
                self.random_seed,
                self.model_path,
                self.epochs,
                self.n_steps,
                self.num_samples_optim,
            )
        return self.currency_model
