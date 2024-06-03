import functions_framework
import pandas as pd
import joblib
import numpy as np
from google.cloud import bigquery, storage
from google.cloud.exceptions import NotFound
from datetime import datetime, timedelta
import os
import settings

def is_running_in_gcp():
    return os.getenv('K_SERVICE', False)

def load_latest_model_and_scaler(bucket_name, gcs_folder_name):
    storage_client = storage.Client() if is_running_in_gcp() else storage.Client.from_service_account_json(settings.CLOUD_STORAGE_CREDENTIALS)
    bucket = storage_client.bucket(bucket_name)
    
    # List all objects in the bucket
    blobs = list(bucket.list_blobs())

    # List all objects in the specific folder
    blobs = list(bucket.list_blobs(prefix=gcs_folder_name))
    
    # Filter and sort blobs by creation time
    model_blobs = sorted([blob for blob in blobs if 'xgboost_model' in blob.name], key=lambda x: x.time_created, reverse=True)
    scaler_blobs = sorted([blob for blob in blobs if 'scaler' in blob.name], key=lambda x: x.time_created, reverse=True)
    feature_names_blobs = sorted([blob for blob in blobs if 'feature_names' in blob.name], key=lambda x: x.time_created, reverse=True)
    
    latest_model_blob = model_blobs[0]
    latest_scaler_blob = scaler_blobs[0]
    latest_feature_names_blob = feature_names_blobs[0]
    
    model_file = f"/tmp/{latest_model_blob.name.split('/')[1]}" #Remove the folder name from the mode_file name. Otherwise, the tmp filepath will generate an error
    scaler_file = f"/tmp/{latest_scaler_blob.name.split('/')[1]}"
    feature_names_file = f"/tmp/{latest_feature_names_blob.name.split('/')[1]}"
    
    latest_model_blob.download_to_filename(model_file)
    latest_scaler_blob.download_to_filename(scaler_file)
    latest_feature_names_blob.download_to_filename(feature_names_file)
    
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    feature_names = joblib.load(feature_names_file)
    
    return model, scaler, feature_names

def load_data_from_bigquery(project_id: str, dataset_id: str, table_name: str) -> pd.DataFrame:
    """Loads data from a BigQuery table into a pandas DataFrame."""
    client = bigquery.Client() if is_running_in_gcp() else bigquery.Client.from_service_account_json(settings.BIGQUERY_CREDENTIALS)

    query = f"""
    SELECT * 
    FROM `{project_id}.{dataset_id}.{table_name}`
    ORDER BY time_open
    """
    df = client.query(query).to_dataframe()
    return df

def get_features_and_target(df):
    """Extracts features and target variable from the DataFrame."""
    X = df.drop(columns=['price_close', 'time_period_start', 'time_period_end', 'time_open', 'time_close'])
    y = df['price_close']
    return X, y

def predict_next_hours(model, scaler, latest_data, train_data, feature_names, hours=50):
    future_predictions = []
    current_time = latest_data['time_period_start'] + pd.Timedelta(hours=1)  # Start from the next hour after the latest data point
    
    for i in range(hours):
        next_hour = current_time + pd.Timedelta(hours=i)
        
        last_24_hours = train_data[train_data['time_period_start'] >= next_hour - pd.Timedelta(hours=24)]
        avg_last_24 = last_24_hours.mean(numeric_only=True)

        last_7_days = train_data[train_data['time_period_start'] >= next_hour - pd.Timedelta(days=7)]
        avg_last_7_days = last_7_days.mean(numeric_only=True)

        next_hour_features = {
            'price_open': latest_data['price_open'],
            'price_high': latest_data['price_high'],
            'price_low': latest_data['price_low'],
            'volume_traded': latest_data['volume_traded'],
            'trades_count': latest_data['trades_count'],
            'price_range': latest_data['price_range'],
            'price_change': latest_data['price_change'],
            'average_price': latest_data['average_price'],
            'volatility': latest_data['volatility'],
            'trading_intensity': latest_data['trading_intensity'],
            'hour': next_hour.hour,
            'day_of_week': next_hour.dayofweek,
            'month': next_hour.month,
            'quarter': next_hour.quarter,
            'price_open_avg_24h': avg_last_24['price_open'],
            'price_high_avg_24h': avg_last_24['price_high'],
            'price_low_avg_24h': avg_last_24['price_low'],
            'volume_traded_avg_24h': avg_last_24['volume_traded'],
            'trades_count_avg_24h': avg_last_24['trades_count'],
            'price_range_avg_24h': avg_last_24['price_range'],
            'price_change_avg_24h': avg_last_24['price_change'],
            'average_price_avg_24h': avg_last_24['average_price'],
            'volatility_avg_24h': avg_last_24['volatility'],
            'trading_intensity_avg_24h': avg_last_24['trading_intensity'],
            'price_open_avg_7d': avg_last_7_days['price_open'],
            'price_high_avg_7d': avg_last_7_days['price_high'],
            'price_low_avg_7d': avg_last_7_days['price_low'],
            'volume_traded_avg_7d': avg_last_7_days['volume_traded'],
            'trades_count_avg_7d': avg_last_7_days['trades_count'],
            'price_range_avg_7d': avg_last_7_days['price_range'],
            'price_change_avg_7d': avg_last_7_days['price_change'],
            'average_price_avg_7d': avg_last_7_days['average_price'],
            'volatility_avg_7d': avg_last_7_days['volatility'],
            'trading_intensity_avg_7d': avg_last_7_days['trading_intensity']
        }
        
        for key, value in next_hour_features.items():
            if pd.isna(value) or np.isinf(value):
                next_hour_features[key] = 0
        
        next_hour_df = pd.DataFrame(next_hour_features, index=[0])
        next_hour_df = next_hour_df[feature_names]  # Ensure columns are in the same order as during training
        next_hour_scaled = scaler.transform(next_hour_df)
        
        predicted_price = model.predict(next_hour_scaled)
        future_predictions.append((next_hour, predicted_price[0]))
        
        # Update the latest_data for the next iteration
        latest_data['price_close'] = predicted_price[0]
        latest_data['time_period_start'] = next_hour
    
    return future_predictions

def upload_df_to_bigquery(dataframe: pd.DataFrame, project_id: str, dataset_id: str, table_name: str, write_mode: str = 'WRITE_APPEND') -> None:
    """Uploads a pandas DataFrame to a BigQuery table."""
    if is_running_in_gcp():
        client = bigquery.Client()
    else:
        client = bigquery.Client.from_service_account_json(settings.BIGQUERY_CREDENTIALS)

    dataset_id = f"{project_id}.{dataset_id}"
    dataset = bigquery.Dataset(dataset_id)
    dataset.location = settings.DATASET_LOCATION
    try:
        dataset = client.create_dataset(dataset, timeout=30)
        print(f"Created dataset {client.project}.{dataset.dataset_id}")
    except Exception as e:
        print(f"Dataset already exists or failed to create: {e}")

    table_id = f"{dataset_id}.{table_name}"
    job_config = bigquery.LoadJobConfig(
        autodetect=True,
        write_disposition=write_mode,
        create_disposition='CREATE_IF_NEEDED'
    )
    try:
        job = client.load_table_from_dataframe(dataframe, table_id, job_config=job_config)
        job.result()
        print("Saved data into BigQuery")
    except Exception as e:
        print(dataframe.dtypes)
        print(table_id)
        print(job_config)
        print(e)
        raise e

def remove_old_predictions(project_id: str, dataset_id: str, table_name: str, new_predictions: pd.DataFrame) -> None:
    """Removes old predictions for the same datetime before inserting new ones."""
    if is_running_in_gcp():
        client = bigquery.Client()
    else:
        client = bigquery.Client.from_service_account_json(settings.BIGQUERY_CREDENTIALS)

    prediction_times = tuple(new_predictions['prediction_time'].astype(str).tolist())
    
    query = f"""
    DELETE FROM `{project_id}.{dataset_id}.{table_name}`
    WHERE prediction_time IN {prediction_times}
    """
    try:
        client.query(query).result()
        print(f"Old predictions removed for times: {prediction_times}")
    except NotFound:
        print("Table not found, proceeding to insert new predictions.")
    except Exception as e:
        print(f"Failed to remove old predictions: {e}")
        raise e

@functions_framework.http
def main(request):
    # Extract the table name from the request
    request_json = request.get_json(silent=True)
    if not request_json or 'bq_table' not in request_json:
        return "BigQuery table name not provided", 400
    bq_table = request_json['bq_table']
    
    # Load the latest model, scaler, and feature names from GCS
    gcs_folder_name = bq_table
    model, scaler, feature_names = load_latest_model_and_scaler(settings.GCS_BUCKET_NAME, gcs_folder_name)

    # Load the data from BigQuery
    df = load_data_from_bigquery(settings.PROJECT_ID, settings.BQ_FEATURE_STORE_DATASET, bq_table)
    if df.empty:
        return "No data found in BigQuery table.", 200

    # Use all available data for training
    train_data = df.copy()

    # Get the latest data point for prediction
    latest_data = train_data.iloc[-1].copy()

    # Predict for the next 10 hours
    future_predictions = predict_next_hours(model, scaler, latest_data, train_data, feature_names)

    # Create a DataFrame for the predictions
    predictions_df = pd.DataFrame({
        'prediction_time': [pred[0] for pred in future_predictions],
        'predicted_price': [pred[1] for pred in future_predictions]
    })

    # Remove old predictions from BigQuery
    remove_old_predictions(settings.PROJECT_ID, settings.BQ_PREDICTIONS_DATASET, bq_table, predictions_df)

    # Save new predictions to BigQuery
    upload_df_to_bigquery(predictions_df, settings.PROJECT_ID, settings.BQ_PREDICTIONS_DATASET, bq_table)
    
    return "Predictions saved successfully!", 200

# The following is used to test the code locally without functions framework (i.e. from Spyder)
if __name__ == '__main__':
    # Mock request for local testing
    class MockRequest:
        def __init__(self, json_data):
            self.json_data = json_data

        def get_json(self, silent=False):
            return self.json_data

    mock_request = MockRequest({"bq_table": "BTC"})

    # Call the function with the mock request
    print(main(mock_request))
