import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)

            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(
            self,
            gender: str,
            age: int,
            membership_type: str,
            country: str,
            join_date: str,
            devices_owned: int,
            profile_count: int,
            avg_watch_minutes: float,
            avg_sessions: float,
            binge_days: int,
            active_days: int,
            late_ratio: float,
            payment_count: int,
            avg_amount_due: float,
            avg_amount_paid: float,
            ticket_count: int,
            avg_resolution: float,
            unresolved_count: int
    ):
        
        self.gender = gender
        self.age = age
        self.membership_type = membership_type
        self.country = country
        self.join_date = join_date
        self.devices_owned = devices_owned
        self.profile_count = profile_count
        self.avg_watch_minutes = avg_watch_minutes
        self.avg_sessions = avg_sessions
        self.binge_days = binge_days
        self.active_days = active_days
        self.late_ratio = late_ratio
        self.payment_count = payment_count
        self.avg_amount_due = avg_amount_due
        self.avg_amount_paid = avg_amount_paid
        self.ticket_count = ticket_count
        self.avg_resolution = avg_resolution
        self.unresolved_count = unresolved_count

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "age": [self.age],
                "membership_type": [self.membership_type],
                "country": [self.country],
                "join_date": [self.join_date],
                "devices_owned": [self.devices_owned],
                "profile_count": [self.profile_count],
                "avg_watch_minutes": [self.avg_watch_minutes],
                "avg_sessions": [self.avg_sessions],
                "binge_days": [self.binge_days],
                "active_days": [self.active_days],
                "late_ratio": [self.late_ratio],
                "payment_count": [self.payment_count],
                "avg_amount_due": [self.avg_amount_due],
                "avg_amount_paid": [self.avg_amount_paid],
                "ticket_count": [self.ticket_count],
                "avg_resolution": [self.avg_resolution],
                "unresolved_count": [self.unresolved_count]
            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)