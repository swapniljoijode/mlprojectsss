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
from src.components.feature_engineering import initiate_feature_engineering


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

    
        
    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            logging.info("Reading train and test data")
            read_train_df = pd.read_csv(train_path)
            read_test_df = pd.read_csv(test_path)
            train_df = initiate_feature_engineering(read_train_df)
            test_df = initiate_feature_engineering(read_test_df)
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
