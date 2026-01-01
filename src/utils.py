import os
import sys
import pandas as pd
import numpy as np
import dill
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def save_object(file_path: str, obj: object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models:dict):
    try:
        report = {}
        for model_name, model in models.items():
            logging.info(f"Training {model_name}")
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_test_pred)
            auc = roc_auc_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred)
            report[model_name] = {'accuracy':accuracy, 'auc':auc, 'f1':f1}
            logging.info(f"{model_name} Model accuracy: {accuracy} ROC_AUC: {auc} F1 Score: {f1}")

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path: str):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)