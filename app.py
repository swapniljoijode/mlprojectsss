from flask import Flask, request, render_template
import pandas as pandas
import numpy as np
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.components.feature_engineering import initiate_feature_engineering
from src.logger import logging

from sklearn.preprocessing import StandardScaler
application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            age=int(request.form.get('age')),
            membership_type=request.form.get('membership_type'),
            country=request.form.get('country'),
            join_date=request.form.get('join_date'),
            devices_owned=int(request.form.get('devices_owned')),
            profile_count=int(request.form.get('profile_count')),
            avg_watch_minutes=float(request.form.get('avg_watch_minutes')),
            avg_sessions=float(request.form.get('avg_sessions')),
            binge_days=int(request.form.get('binge_days')),
            active_days=int(request.form.get('active_days')),
            late_ratio=float(request.form.get('late_ratio')),
            payment_count=int(request.form.get('payment_count')),
            avg_amount_due=float(request.form.get('avg_amount_due')),
            avg_amount_paid=float(request.form.get('avg_amount_paid')),
            ticket_count=int(request.form.get('ticket_count')),
            avg_resolution=float(request.form.get('avg_resolution')),
            unresolved_count=int(request.form.get('unresolved_count'))
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        logging.info("Obtained input data as dataframe")
        pred_df = initiate_feature_engineering(pred_df)
        logging.info("Feature engineering completed on input data")
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        logging.info("Prediction completed")
        logging.info(f"Prediction results: {results}, type: {type(results)}, shape: {np.array(results).shape}, values: {(results[0])}")
        return render_template('home.html', results=int(results[0]))
    
if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)