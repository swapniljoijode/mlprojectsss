import sys
import pandas as pd
import os
import numpy as np
from src.logger import logging
from src.exception import CustomException


def initiate_feature_engineering(df:pd.DataFrame) :
        try:
            
            logging.info("dataset for feature engineering read successfully")

            logging.info("Starting feature engineering")
            
            df["join_date"] = pd.to_datetime(df["join_date"])
            df["tenure_days"] = (pd.Timestamp("2024-12-31") - df["join_date"]).dt.days
            # df["watch_bucket"] = pd.qcut(
            #     df["avg_watch_minutes"],
            #     q=5,
            #     labels=["Very Low", "Low", "Medium", "High", "Very High"]
            # )
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