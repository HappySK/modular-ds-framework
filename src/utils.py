import os
import sys
import numpy as np
import pandas as pd
import dill, yaml, argparse
import mlflow

from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException('Invalid Save Config', e, sys)

def evaluate_model(X_train, X_test, y_train, y_test, models, params):
    try:
        report = {}
        config = get_yaml_config("src/config/model_config.yaml")

        for model_nm, model_class in models.items():
            mlflow.set_experiment(f"STUD_PERF_PRED_{model_nm.upper()}")
            para = params[model_nm]

            gs = GridSearchCV(model_class, para, cv=3)
            gs.fit(X_train, y_train)

            model_class.set_params(**gs.best_params_)
            model_class.fit(X_train, y_train)

            logging.info(f"Evaluating the model - {model_nm}")
            y_test_pred = model_class.predict(X_test)

            metrics = get_metrics(y_test, y_test_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_nm] = test_model_score

            with mlflow.start_run():
                mlflow.log_params(para)
                mlflow.log_metrics(metrics)
                mlflow.set_tags(config["mlflow"]["tags"])

        return report

    except Exception as e:
        logging.info("Invalid Evaluate Model")
        raise CustomException("Invalid Evaluate Model", e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException("Invalid Load Value", e, sys)


def get_yaml_config(file_name="src/config/model_config.yaml"):

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    dataset = args.dataset

    with open(file_name) as stream:
        input_config = yaml.safe_load(stream)
    return input_config[dataset]

def get_columns(file_name="src/config/model_config.yaml"):
    config = get_yaml_config(file_name)

    numerical_columns = config["columns"]["numerical"]
    categorical_columns = config["columns"]["categorical"]
    target_column = config["columns"]["target"]

    return numerical_columns, categorical_columns, target_column

def get_metrics(pred_val, act_val):
    config = get_yaml_config()
    metrics = {}
    for metric in config["mlflow"]["metrics"]:
        metrics[metric] = eval(metric)(act_val, pred_val)

    return metrics