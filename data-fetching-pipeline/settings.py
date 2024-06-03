from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# API Settings
API_KEY = os.getenv('API_KEY')

# GCP Settings
PROJECT_ID = os.getenv('PROJECT_ID')
BUCKET_NAME = f"{PROJECT_ID}_raw_crypto_data"
DATASET_ID = "historical_crypto_prices"

# Time Settings
END_DATE = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
START_DATE = END_DATE - timedelta(days=10 * 365)

# Service Account JSON files
BIGQUERY_CREDENTIALS = os.getenv('BIGQUERY_CREDENTIALS')
STORAGE_CREDENTIALS = os.getenv('STORAGE_CREDENTIALS')

# GCP Locations
DATASET_LOCATION = 'europe-west8'