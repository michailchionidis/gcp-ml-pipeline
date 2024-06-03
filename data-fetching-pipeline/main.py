import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
from google.cloud import storage, bigquery
import json
from flask import Flask, request, jsonify
import settings

app = Flask(__name__)

def is_running_in_gcp():
    return os.getenv('K_SERVICE', False)
    
def fetch_crypto_data(symbol: str, start_date: str, end_date: str, api_key: str) -> list:
    """Fetches hourly cryptocurrency data from CoinAPI."""
    url = f'https://rest.coinapi.io/v1/ohlcv/{symbol}/history'
    headers = {"X-CoinAPI-Key": api_key}
    params = {
        "period_id": "1HRS",
        "time_start": start_date,
        "time_end": end_date,
        "limit": 100000
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

def get_latest_timestamp(bigquery_client: bigquery.Client, bigquery_table_id: str) -> datetime:
    query = f"""
        SELECT MAX(time_period_start) as last_timestamp
        FROM `{bigquery_table_id}`
    """
    query_job = bigquery_client.query(query)
    results = query_job.result()
    for row in results:
        if row.last_timestamp:
            timestamp_str = row.last_timestamp.split('.')[0]
            return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    return None

def upload_json_to_gcs(json_data: dict, bucket_name: str, folder_path: str, symbol: str, start_date: str, end_date: str) -> None:
    """Uploads a JSON object to Google Cloud Storage."""
    if is_running_in_gcp():
        client = storage.Client()
    else:
        client = storage.Client.from_service_account_json(settings.STORAGE_CREDENTIALS)

    try:
        bucket = client.get_bucket(bucket_name)
    except:
        bucket = client.create_bucket(bucket_name)
    filename = f"{symbol}_{start_date}_{end_date}.json"
    object_path = os.path.join(folder_path, filename)
    blob = bucket.blob(object_path)
    blob.upload_from_string(json.dumps(json_data).encode("utf-8"), content_type="application/json")
    print(f"Uploaded {filename} to {bucket_name}/{object_path}")

def upload_df_to_bigquery(dataframe: pd.DataFrame, project_id: str, dataset_id: str, table_name: str) -> None:
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
    except:
        print("Dataset already exists")
    table_id = f"{dataset_id}.{table_name}"
    job_config = bigquery.LoadJobConfig(
          autodetect=True,
          write_disposition='WRITE_APPEND',
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

@app.route('/', methods=['POST'])
def main(request):
    request_data = request.get_json()
    symbol = request_data.get('symbol')
    crypto_code = request_data.get('crypto_code')

    if not symbol or not crypto_code:
        return jsonify({"error": "Missing 'symbol' or 'crypto_code' in request"}), 400

    api_key = settings.API_KEY
    project_id = settings.PROJECT_ID
    bucket_name = settings.BUCKET_NAME
    folder_path = crypto_code
    dataset_id = settings.DATASET_ID
    table_name = crypto_code
    end_date = settings.END_DATE
    start_date = settings.START_DATE

    if is_running_in_gcp():
        bigquery_client = bigquery.Client()
    else:
        bigquery_client = bigquery.Client.from_service_account_json(settings.BIGQUERY_CREDENTIALS)

    try:
        latest_timestamp = get_latest_timestamp(bigquery_client, f"{project_id}.{dataset_id}.{table_name}")
    except Exception as e:
        print(e)
        latest_timestamp = None
        

    if latest_timestamp:
        print(f'There are already historical data saved in BigQuery until {latest_timestamp} for the selected Cryptocurrency')
        start_date = latest_timestamp + timedelta(hours=1)
        if start_date >= end_date:
            print("No new data to fetch. The latest data is already stored in BigQuery.")
            return jsonify({"message": "No new data to fetch. The latest data is already stored in BigQuery."}), 200
    else:
        print('Start retrieving historical data for the past 10 years')
        start_date = end_date - timedelta(days=365*10)

    all_data = []
    while start_date < end_date:
        chunk_end_date = min(start_date + timedelta(days=365), end_date)
        data = fetch_crypto_data(symbol, start_date.isoformat(), chunk_end_date.isoformat(), api_key)
        all_data.extend(data)
        print(f"Extracted {symbol} historical data from {start_date} to {chunk_end_date}")
        start_date = chunk_end_date

    if not all_data:
        print("No data found for the specified cryptocurrency")
        return jsonify({"message": "No data found for the specified cryptocurrency"}), 200

    upload_json_to_gcs(all_data, bucket_name, folder_path, symbol, (end_date - timedelta(days=365*10)).isoformat(), end_date.isoformat())
    df = pd.DataFrame(all_data)
    upload_df_to_bigquery(df, project_id, dataset_id, table_name)
    print("Data extracted, loaded into BigQuery, and saved to Cloud Storage successfully")
    return jsonify({"message": "Data extracted, loaded into BigQuery, and saved to Cloud Storage successfully"}), 200

if __name__ == "__main__":
    app.run(debug=True)

