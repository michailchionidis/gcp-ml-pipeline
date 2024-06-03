import functions_framework
import pandas as pd
from google.cloud import bigquery
import os
import settings
from datetime import datetime, timezone
from google.api_core.exceptions import NotFound

def is_running_in_gcp():
    return os.getenv('K_SERVICE', False)

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

def get_latest_time_period_start(project_id: str, dataset_id: str, table_name: str) -> str:
    """Gets the latest time_period_start from the destination table."""
    if is_running_in_gcp():
        client = bigquery.Client()
    else:
        client = bigquery.Client.from_service_account_json(settings.BIGQUERY_CREDENTIALS)

    query = f"""
    SELECT MAX(time_period_start) as latest_time_period_start
    FROM `{project_id}.{dataset_id}.{table_name}`
    """
    try:
        result = client.query(query).result()
        for row in result:
            if row.latest_time_period_start:
                return row.latest_time_period_start.strftime("%Y-%m-%dT%H:%M:%S")
    except NotFound:
        print(f"Table {project_id}.{dataset_id}.{table_name} not found. Fetching all data from source.")
        return None

def get_historical_data_from_source_table(bq_source_table, latest_time_period_start):
    """Loads historical cryptocurrency data from the BigQuery source table"""
    if is_running_in_gcp():
        bq_client = bigquery.Client()
    else:
        bq_client = bigquery.Client.from_service_account_json(settings.BIGQUERY_CREDENTIALS)

    # Query new data from the source BigQuery table
    query = f"""
    SELECT * FROM `{settings.PROJECT_ID}.{settings.BQ_SOURCE_DATASET}.{bq_source_table}`
    WHERE PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%E*SZ', time_period_start) > TIMESTAMP('{latest_time_period_start}')
    ORDER BY time_open;
    """
    df = bq_client.query(query).to_dataframe()
    return df

def feature_engineering(df):
    """Performs feature engineering on the data."""
    
    # Convert time_period_start and other time columns to datetime
    df['time_period_start'] = pd.to_datetime(df['time_period_start'], utc=True)
    df['time_period_end'] = pd.to_datetime(df['time_period_end'], utc=True)
    df['time_open'] = pd.to_datetime(df['time_open'], utc=True)
    df['time_close'] = pd.to_datetime(df['time_close'], utc=True)

    # Extract seasonality features based on the time_period_start
    df['hour'] = df['time_period_start'].dt.hour  # Extract hour of the day (0-23)
    df['day_of_week'] = df['time_period_start'].dt.dayofweek  # Extract day of the week (0=Monday, 6=Sunday)
    df['month'] = df['time_period_start'].dt.month  # Extract month (1-12)
    df['quarter'] = df['time_period_start'].dt.quarter  # Extract quarter (1-4)

    # Process original features to generate more features
    df['price_range'] = df['price_high'] - df['price_low']
    df['price_change'] = df['price_close'] - df['price_open']
    df['average_price'] = (df['price_high'] + df['price_low'] + df['price_close']) / 3
    df['volatility'] = df[['price_open', 'price_high', 'price_low', 'price_close']].std(axis=1)
    df['trading_intensity'] = df['volume_traded'] / df['trades_count']

    # Calculate 24-hour rolling average features
    df['price_open_avg_24h'] = df['price_open'].rolling(window=24, min_periods=1).mean()  # 24-hour rolling average of opening price
    df['price_high_avg_24h'] = df['price_high'].rolling(window=24, min_periods=1).mean()  # 24-hour rolling average of highest price
    df['price_low_avg_24h'] = df['price_low'].rolling(window=24, min_periods=1).mean()  # 24-hour rolling average of lowest price
    df['volume_traded_avg_24h'] = df['volume_traded'].rolling(window=24, min_periods=1).mean()  # 24-hour rolling average of traded volume
    df['trades_count_avg_24h'] = df['trades_count'].rolling(window=24, min_periods=1).mean()  # 24-hour rolling average of trades count
    df['price_range_avg_24h'] = df['price_range'].rolling(window=24, min_periods=1).mean()  # 24-hour rolling average of price range (high - low)
    df['price_change_avg_24h'] = df['price_change'].rolling(window=24, min_periods=1).mean()  # 24-hour rolling average of price change (close - open)
    df['average_price_avg_24h'] = df['average_price'].rolling(window=24, min_periods=1).mean()  # 24-hour rolling average of average price
    df['volatility_avg_24h'] = df['volatility'].rolling(window=24, min_periods=1).mean()  # 24-hour rolling average of price volatility (standard deviation)
    df['trading_intensity_avg_24h'] = df['trading_intensity'].rolling(window=24, min_periods=1).mean()  # 24-hour rolling average of trading intensity (volume / trades count)

    # Calculate 7-day rolling average features
    df['price_open_avg_7d'] = df['price_open'].rolling(window=24*7, min_periods=1).mean()  # 7-day rolling average of opening price
    df['price_high_avg_7d'] = df['price_high'].rolling(window=24*7, min_periods=1).mean()  # 7-day rolling average of highest price
    df['price_low_avg_7d'] = df['price_low'].rolling(window=24*7, min_periods=1).mean()  # 7-day rolling average of lowest price
    df['volume_traded_avg_7d'] = df['volume_traded'].rolling(window=24*7, min_periods=1).mean()  # 7-day rolling average of traded volume
    df['trades_count_avg_7d'] = df['trades_count'].rolling(window=24*7, min_periods=1).mean()  # 7-day rolling average of trades count
    df['price_range_avg_7d'] = df['price_range'].rolling(window=24*7, min_periods=1).mean()  # 7-day rolling average of price range (high - low)
    df['price_change_avg_7d'] = df['price_change'].rolling(window=24*7, min_periods=1).mean()  # 7-day rolling average of price change (close - open)
    df['average_price_avg_7d'] = df['average_price'].rolling(window=24*7, min_periods=1).mean()  # 7-day rolling average of average price
    df['volatility_avg_7d'] = df['volatility'].rolling(window=24*7, min_periods=1).mean()  # 7-day rolling average of price volatility (standard deviation)
    df['trading_intensity_avg_7d'] = df['trading_intensity'].rolling(window=24*7, min_periods=1).mean()  # 7-day rolling average of trading intensity (volume / trades count)

    # Drop any rows with missing values that resulted from the rolling window calculations
    df = df.dropna()
    return df


@functions_framework.http
def main(request):
    
    # Extract the source table name from the request
    request_json = request.get_json(silent=True)
    if not request_json or 'bq_table' not in request_json:
        return "BigQuery table name not provided", 400
    bq_table = request_json['bq_table']

    # Get the latest time_period_start from the destination table (feature store)
    latest_time_period_start = get_latest_time_period_start(settings.PROJECT_ID, settings.BQ_FEATURE_STORE_DATASET, bq_table)
    if latest_time_period_start is None:
        latest_time_period_start = '1970-01-01T00:00:00'  # If no data exists, start from the earliest possible date
    
    df = get_historical_data_from_source_table(bq_table, latest_time_period_start)

    if df.empty:
        return "No new data to process.", 200

    # Perform feature engineering
    df = feature_engineering(df)

    # Select and order the columns to store
    features = df[['time_period_start', 'time_period_end', 'time_open', 'time_close', 'price_open',
                   'price_high', 'price_low', 'price_close', 'volume_traded', 'trades_count', 'price_range', 
                   'price_change', 'average_price', 'volatility', 'trading_intensity',
                   'price_range_avg_24h', 'price_change_avg_24h', 'average_price_avg_24h', 'volatility_avg_24h', 
                   'trading_intensity_avg_24h', 'price_open_avg_24h', 'price_high_avg_24h', 
                   'price_low_avg_24h', 'volume_traded_avg_7d', 'trades_count_avg_7d', 
                   'price_range_avg_7d', 'price_change_avg_7d', 'average_price_avg_7d', 
                   'volatility_avg_7d', 'trading_intensity_avg_7d', 'hour', 'day_of_week', 
                   'month', 'quarter']].dropna()

    # Save features back to BigQuery (Feature Store)
    upload_df_to_bigquery(dataframe=features, project_id=settings.PROJECT_ID, dataset_id=settings.BQ_FEATURE_STORE_DATASET, table_name=bq_table, write_mode='WRITE_APPEND')

    return "Features stored successfully!", 200

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
