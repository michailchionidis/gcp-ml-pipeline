from dotenv import load_dotenv
import os
from datetime import datetime, timedelta, timezone

# Load environment variables from .env file
load_dotenv()

# Project settings
PROJECT_ID = os.getenv('PROJECT_ID')
BIGQUERY_CREDENTIALS = os.getenv('BIGQUERY_CREDENTIALS')

# BigQuery settings
BQ_SOURCE_DATASET = "historical_crypto_prices"
BQ_FEATURE_STORE_DATASET = "cryptocurrency_feature_store"

# GCP Locations
DATASET_LOCATION = 'europe-west8'