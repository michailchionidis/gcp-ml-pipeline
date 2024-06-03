#!/bin/bash
# Load environment variables from .env file
export $(grep -v '^#' .env | xargs)

# Set your project ID and region
PROJECT_ID=$PROJECT_ID
REGION=$REGION
CLOUD_WORKFLOW_INVOCATION_URL=$CLOUD_WORKFLOW_INVOCATION_URL
SERVICE_ACCOUNT_EMAIL=$SERVICE_ACCOUNT_EMAIL

# Set the crypto code and symbol:
CRYPTO_CODE="BTC"
SYMBOL="BITSTAMP_SPOT_BTC_USD"

# Create Cloud Scheduler job for data fetching and inference workflow
gcloud scheduler jobs create http end-to-end-pipeline-scheduler-$CRYPTO_CODE \
  --schedule "0 * * * *" \
  --http-method POST \
  --uri "${CLOUD_WORKFLOW_INVOCATION_URL}" \
  --oauth-service-account-email "${SERVICE_ACCOUNT_EMAIL}" \
  --oauth-token-scope "https://www.googleapis.com/auth/cloud-platform" \
  --message-body "{\"argument\":\"{\\\"crypto_code\\\": \\\"${CRYPTO_CODE}\\\", \\\"symbol\\\": \\\"${SYMBOL}\\\"}\",\"callLogLevel\":\"CALL_LOG_LEVEL_UNSPECIFIED\"}" \
  --time-zone "UTC" \
  --location ${REGION} \
  --headers "Content-Type=application/json"
  
