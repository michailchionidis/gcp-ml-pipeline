import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Replace with the actual URL of your Cloud Function
url = os.getenv('CLOUD_FUNCTION_TRIGGER_URL') 

# Add data to be sent in the POST request body (as a dictionary)
# In this case, we should specify the name of the source BigQuery table where the features have been stored.
data = {"bq_table": "BTC"} # Replace with your BigQuery table name

# Send the POST request
response = requests.post(url, json=data)  # Use json=data if sending data

# Check for successful response
if response.status_code == 200:
  print("POST request successful!")
  print(response.text)
  # Access the response content (optional)
  # print(response.text)
else:
  print(f"Error: {response.status_code}")
  print(response.text)