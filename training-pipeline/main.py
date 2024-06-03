import functions_framework
import pandas as pd
from google.cloud import bigquery, storage
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib
import os
import settings
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from datetime import datetime
import warnings
import tempfile

# Suppress warnings
warnings.filterwarnings("ignore")

def is_running_in_gcp():
    return os.getenv('K_SERVICE', False)

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

def train_model(X_train, y_train):
    """Trains a model and returns the best model and scaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [2, 3, 4],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.8, 1.0],  # Add subsampling to reduce overfitting
        'colsample_bytree': [0.8, 1.0]  # Add column subsampling to reduce overfitting
    }
    xgb = XGBRegressor(random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    
    return best_model, scaler, X_train.columns.tolist()  # Return feature names

def evaluate_model(best_model, scaler, X_train, y_train, X_test, y_test, dates_train, dates_test):
    """Evaluates the model and checks for overfitting."""
    # Scale the test set
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Predict on train and test sets
    y_train_pred = best_model.predict(X_train_scaled)
    y_test_pred = best_model.predict(X_test_scaled)
    
    # Calculate MAE
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f'Training MAE: {train_mae}, Test MAE: {test_mae}')
    
    # Plot actual vs predicted prices
    plt.figure(figsize=(14, 7))
    plt.plot(dates_train, y_train.values, label='Actual Prices (Train)', color='blue')
    plt.plot(dates_train, y_train_pred, label='Predicted Prices (Train)', color='red', linestyle='--')
    plt.plot(dates_test, y_test.values, label='Actual Prices (Test)', color='green')
    plt.plot(dates_test, y_test_pred, label='Predicted Prices (Test)', color='orange', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Prices')
    plt.legend()
    
    plot_filename = os.path.join(tempfile.gettempdir(), 'actual_vs_predicted_prices.png')
    
    # Save the plot locally
    plt.savefig(plot_filename)
    plt.close()

    return plot_filename

def upload_to_gcs(bucket_name, gcs_folder_name, file_paths):
    """Uploads files to Google Cloud Storage."""
    storage_client = storage.Client() if is_running_in_gcp() else storage.Client.from_service_account_json(settings.CLOUD_STORAGE_CREDENTIALS)
    
    try:
        bucket = storage_client.get_bucket(bucket_name)
    except Exception as e:
        print(f"Creating bucket {bucket_name} as it does not exist. Error: {e}")
        bucket = storage_client.create_bucket(bucket_name)
    
    # Generate a unique identifier using the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    for file_path in file_paths:
        filename_with_timestamp = f"{gcs_folder_name}/{os.path.basename(file_path).split('.')[0]}_{timestamp}.{os.path.basename(file_path).split('.')[-1]}"
        
        # Upload file
        blob = bucket.blob(filename_with_timestamp)
        blob.upload_from_filename(file_path)
        print(f"Uploaded {filename_with_timestamp} to {bucket_name}")

@functions_framework.http
def main(request):
    # Extract the table name from the request
    request_json = request.get_json(silent=True)
    if not request_json or 'bq_table' not in request_json:
        return "BigQuery table name not provided", 400
    bq_table = request_json['bq_table']

    # Load the data from BigQuery
    df = load_data_from_bigquery(settings.PROJECT_ID, settings.BQ_FEATURE_STORE_DATASET, bq_table)

    if df.empty:
        return "No data found in BigQuery table.", 200
    
    # Split features and target
    X, y = get_features_and_target(df)
    
    # Get date indices for plotting
    dates = pd.to_datetime(df['time_period_start'], utc=True)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X, y, dates, test_size=0.2, random_state=42, shuffle=False)
    
    # Train the model
    best_model, scaler, feature_names = train_model(X_train, y_train)
    
    # Evaluate the model and check for overfitting
    plot_filename = evaluate_model(best_model, scaler, X_train, y_train, X_test, y_test, dates_train, dates_test)
    
    # Save the model, scaler, and feature names locally
    model_filename = os.path.join(tempfile.gettempdir(), 'xgboost_model.pkl')
    scaler_filename = os.path.join(tempfile.gettempdir(), 'scaler.pkl')
    feature_names_filename = os.path.join(tempfile.gettempdir(), 'feature_names.pkl')
    
    joblib.dump(best_model, model_filename)
    joblib.dump(scaler, scaler_filename)
    joblib.dump(feature_names, feature_names_filename)
    print("Model, scaler, and feature names saved locally.")
    
    # Upload the model, scaler, and plot to GCS
    gcs_folder_name = bq_table
    upload_to_gcs(settings.GCS_BUCKET_NAME, gcs_folder_name, [model_filename, scaler_filename, feature_names_filename, plot_filename])
    
    return "Model trained and saved successfully!", 200

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
