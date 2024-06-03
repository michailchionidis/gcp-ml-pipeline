# Load environment variables from .env file
export $(grep -v '^#' .env | xargs)

gcloud functions deploy extract_crypto_data \
    --region=europe-west2 \
    --source=. \
    --entry-point=main \
    --runtime=python311 \
    --trigger-http \
    --allow-unauthenticated \
    --timeout=540 \
    --memory=512MB \
    --set-env-vars PROJECT_ID=$PROJECT_ID,API_KEY=$API_KEY