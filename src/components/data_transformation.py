import sys
from dataclasses import dataclass
import numpy as np
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        

    def get_data_transformer_object(self,numericalcolumns,categoricalcolumns):
        try:
            logging.info(f"Data Transformation initiated on {numericalcolumns} and {categoricalcolumns}")
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info(f"Numerical columns imputing and scaling completed")

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder())
                ]
            )
            logging.info(f"Categorical columns imputing and encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numericalcolumns),
                    ("cat_pipeline", cat_pipeline, categoricalcolumns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_feature_engineering(self, df_path: str):
        try:
            df = pd.read_csv(df_path)
            logging.info("Read train and test data completed")

            logging.info("Starting feature engineering")
            
            df["join_date"] = pd.to_datetime(df["join_date"])
            df["tenure_days"] = (pd.Timestamp("2024-12-31") - df["join_date"]).dt.days
            df["watch_bucket"] = pd.qcut(
                df["avg_watch_minutes"],
                q=5,
                labels=["Very Low", "Low", "Medium", "High", "Very High"]
            )
            df["late_bucket"] = pd.cut(
                df["late_ratio"],
                bins=[0, 0.05, 0.15, 0.3, 1.0],
                labels=["No Issues", "Minor", "Moderate", "Severe"]
            )
            df["low_usage"] = (df["avg_watch_minutes"] < 60).astype(int)
            df["payment_issue"] = (df["late_ratio"] > 0.1).astype(int)



            logging.info("Feature engineering completed")

            # Save the engineered features back to CSV if needed
            return df
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            train_df = self.initiate_feature_engineering(train_path)
            test_df = self.initiate_feature_engineering(test_path)
            logging.info("Read train and test data completed")

            numericalcolumns = train_df.drop(columns=['churn']).select_dtypes(include=['int64', 'float64']).columns.tolist()
            categoricalcolumns = train_df.drop(columns=['customer_id']).select_dtypes(include=['object', 'category']).columns.tolist()
            logging.info("Obtaining preprocessor object")

            preprocessing_obj = self.get_data_transformer_object(numericalcolumns=numericalcolumns, categoricalcolumns=categoricalcolumns)

            logging.info("Applying preprocessing object on training and testing datasets") 

            target_column_name = "churn"
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)



            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
