import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Replace with the actual URL of your Cloud Function
url = os.getenv('CLOUD_FUNCTION_TRIGGER_URL') 

# Data to be sent in the POST request body
data = {
    "symbol": "BITSTAMP_SPOT_BTC_USD",
    "crypto_code": "BTC"
}

# Send the POST request
response = requests.post(url, json=data)

# Check for successful response
if response.status_code == 200:
    print("POST request successful!")
    print(response.json())
else:
    print(f"Error: {response.status_code}")
    print(response.text)