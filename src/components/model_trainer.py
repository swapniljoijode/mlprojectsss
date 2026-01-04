import pandas as pd
import sys
import os
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "XGBoost": XGBClassifier(eval_metric='logloss'),
                "CatBoost": CatBoostClassifier(verbose=0),
                "Decision Tree": DecisionTreeClassifier()
            }

            model_report:dict = evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models)

            

            best_model_name = max(model_report, key=lambda x: (model_report[x]['accuracy'] + model_report[x]['auc'] + model_report[x]['f1']) / 3)
            best_model = models[best_model_name]
            best_model_score = model_report[best_model_name]['accuracy']
            logging.info(f"Best model found: {best_model_name} with ROC AUC: {model_report[best_model_name]['auc']} and F1 Score: {model_report[best_model_name]['f1']}")



            if best_model_score < 0.6:
                raise CustomException("No best model found with accuracy greater than 0.6", sys)
            logging.info("Best model on both training and testing dataset found")


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(X_test)
            final_score = f1_score(y_test, predicted)
            return final_score

        except Exception as e:
            raise CustomException(e, sys)
    

