# End-to-End Machine Learning Pipeline on Google Cloud Platform:

## Automated Cryptocurrency Price Forecasting:
This repository showcases an automated cryptocurrency forecasting pipeline developed to predict future cryptocurrency prices. The project demonstrates the setup of a complete machine learning pipeline on Google Cloud Platform (GCP).

## Project Architecture

![Project Architecture](https://github.com/michailchionidis/gcp-ml-pipeline/assets/104796421/4b5684a5-84db-480e-8d02-20ac3ab220dc)


## Skills Demonstrated - Building Blocks of the Machine Learning Pipeline

- **Data Fetching Pipeline:**
  - **Data Extraction:** Utilizes a Cloud Function to extract raw cryptocurrency data from CoinAPI and store it in BigQuery and Cloud Storage.

- **Machine Learning Pipeline:**
  - **Feature Engineering:** Implements a Cloud Function to process raw data into meaningful features stored in BigQuery.
  - **Model Training:** Automates the training process using a Cloud Function. This includes feature extraction from BigQuery, model training with XGBoost, and storing the trained model and scaler in Cloud Storage, effectively using it as a model registry.
  - **Inference:** Deploys a Cloud Function to use the trained models for generating hourly predictions of future cryptocurrency prices.

- **Job Orchestration and Automation:**
  - **Cloud Workflows:** Orchestrates the sequential execution of data fetching, feature engineering, training, and inference pipelines.
  - **Cloud Scheduler:** Automates the execution schedule, ensuring the pipeline runs hourly and the model is retrained daily.

- **Data Visualization:**
  - **Looker Studio:** Visualizes predictions in an interactive dashboard, providing up-to-date data on cryptocurrency prices.

## Technologies Used

- **Python**
- **Google Cloud Platform:** Cloud Functions, Cloud Storage, BigQuery, Cloud Workflows, Cloud Scheduler, Looker Studio
- **Machine Learning Libraries:** scikit-learn, XGBoost

## Project Files

- **data-fetching-pipeline/main.py:** Cloud Function script for extracting raw cryptocurrency data.
- **feature-pipeline/main.py:** Cloud Function script for feature engineering.
- **training-pipeline/main.py:** Cloud Function script for training the machine learning model.
- **inference-pipeline/main.py:** Cloud Function script for generating predictions.
- **cloud-workflows/main.yaml:** YAML file for defining the workflow to orchestrate the pipelines.
- **settings.py:** Configuration settings for API keys and database connections.
- **requirements.txt:** Lists all Python dependencies.
- **deploy.sh:** Shell script for deploying the Cloud Functions and Workflows.

## Setup and Installation

### Prerequisites

- A Google Cloud Platform account. Start your free trial [here](https://cloud.google.com/free).
- Basic knowledge of Python and machine learning concepts.
- Familiarity with Google Cloud services.

### Configuration and Deployment

1. **Clone the repository**

    ```bash
    git clone https://github.com/yourusername/crypto-forecasting-pipeline.git
    ```

2. **Set up Google Cloud Platform services**

    - Create a new project in Google Cloud Platform
    - Enable the required APIs: Cloud Functions, Cloud Storage, BigQuery, Cloud Scheduler, Workflows
    - Create service accounts with the following permissions:
        - Read/Write to BigQuery
        - Read/Write to Cloud Storage
        - Workflows invoker
    - Generate keys (.JSON files) for those service accounts and download them locally.

3. **Test the Cloud Functions locally**

    - Navigate to each pipeline directory (`data-fetching-pipeline`, `feature-pipeline`, `training-pipeline`, `inference-pipeline`).
    - Create a virtual environment: `python3 -m venv venv` and activate it: `source venv/bin/activate`
    - Install dependencies: `pip install -r requirements.txt`
    - Create a `.env` file and fill in environment variables based on the corresponding `settings.py` file of the same directory
    - Run the respective `main.py` scripts locally to validate functionality.

4. **Deploy on Google Cloud**

    - Use the provided `deploy.sh` script to deploy the Cloud Functions and Workflows.
    - Set up Cloud Scheduler to automate the execution of the pipelines.

### Detailed Instructions

Detailed project implementation and deployment instructions are provided by DataProjects.io in the project material, including how to:

- Set up Google Cloud Platform project and services.
- Build the data and machine learning pipelines
- Deploy and monitor the automated execution of the end-to-end project

## Running the Project

1. Ensure all settings in `settings.py` are correctly configured.
2. Deploy the Cloud Functions and Workflows to Google Cloud.
3. Use Cloud Scheduler to manage the pipeline execution schedule.

## Contributions

Contributions to this project are welcome! Please fork this repository and submit a pull request with your proposed changes.

## Acknowledgments

This project is provided by [DataProjects.io](https://dataprojects.io), a platform that helps data professionals build a portfolio of real-world, end-to-end projects on the cloud.

## License

This project is licensed under the Mozilla Public License 2.0 - see the LICENSE file for details.
