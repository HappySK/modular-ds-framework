import os
import sys
import numpy as np
import pandas as pd
import dill, yaml, argparse

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

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

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            logging.info("Evaluating the model")
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

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


def get_yaml_config(file_name):

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