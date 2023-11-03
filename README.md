# Diabetes Predictor: Early Detection for Healthier Lives

## Introduction 🚀
Welcome to our Diabetes Prediction project! 📈
Our project focuses on using machine learning to predict diabetes, a common health condition. Predicting diabetes early can be really helpful in managing it effectively.
In this guide, we'll walk you through how our project works, so you can understand, use, and even contribute to it. Together, we can make a positive impact on healthcare and well-being! 

## Requirements 📄
+ Python 3.8 and above
+ Required python libraries - evidently, fastapi, imblearn, joblib, mlflow, numpy, pandas, prefect, scikit-learn, streamlit, uvicorn, xgboost
+ The Diabetes Prediction datset is available at [click here](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_binary_health_indicators_BRFSS2015.csv)

## Getting started
* Navigate to the Directory Where You Want to Clone the Repository:
Use the cd (change directory) command to navigate to the directory where you want to clone the repository. For example:
'''bash
cd /path/to/your/desired/directory
'''
* Clone the Repository:
To clone the "Diabetes-prediction-MLOps" repository, you need to use the following command.
```bash
git clone https://github.com/LohithUnnam/Diabetes-prediction-MLOps.git
```


## Project Workflow: Building a Diabetes Prediction Pipeline
Our project, "Diabetes Predictor: Early Detection for Healthier Lives," is dedicated to providing accurate diabetes predictions through a meticulously designed pipeline.

### Reliability in Predictions
In this project we've established an end-to-end workflow that encompasses data ingestion, data preprocessing, model development, model evaluation, and experiment tracking. This orchestration is driven by Prefect, streamlining the flow of tasks.

### Experiment Tracking
Experiment tracking is the final step in our diabetes prediction workflow. It is responsible for logging important information about the trained machine learning model and the results of our predictions. We utilize MLflow for experiment tracking, which includes logging hyperparameters, metrics, and artifacts.

### Key Workflow Steps
- Data Ingestion (ingest_data): Collect and structure data into a DataFrame for analysis.

- Data Preprocessing (clean_data): Clean and prepare the data for modeling.

- Model Development (train_model): Train the machine learning model with autologging.

- Model Evaluation (evaluation): Assess the model's performance and log key metrics.

- Experiment Tracking (track_experiment): Log hyperparameters, metrics, and artifacts using MLflow for transparency and model management.

This streamlined workflow ensures efficient diabetes prediction and transparent model monitoring for proactive healthcare.

To run this workflow run the following command:
```python
python execute_flow.py
```
To track your workflows in server run the below command:
```python
prefect server start
```
After executing the command and navigating to the provided server link, you can view all your flow runs, both successful and failed, along with the execution times for each task and the entire workflow.

![Image](https://docs.prefect.io/2.14.3/img/ui/cloud-dashboard.png)

## Model Deployment with FastAPI
In our project, we have successfully deployed our diabetes prediction model using FastAPI, a modern, fast, and web-based framework for building APIs. This deployment allows healthcare professionals and individuals to access our model for making diabetes risk assessments.

### Key Components:

FastAPI Setup: We've created a FastAPI application, defining routes and handling incoming requests for predictions.

Pre-trained Models: Our application loads pre-trained machine learning models, specifically Random Forest Classifier (rfc) and Decision Tree Classifier (dtc), from saved model files.

Data Input: We accept input data that represents various health and lifestyle attributes using the Pydantic library to validate and structure the input.

Model Selection: Users can specify the model they want to use for prediction, ensuring flexibility in choosing the appropriate model.

Data Preprocessing: Input data is preprocessed, with specific columns scaled using a pre-trained StandardScaler to align with the model's requirements.

Prediction: The selected model is used to predict diabetes risk based on the provided input data.

#### API Endpoints:

/predict/: This endpoint is responsible for receiving POST requests with input data and model selection, and it returns a prediction indicating the likelihood of diabetes (binary prediction).

/: A simple welcome endpoint, providing a friendly greeting to users.

This FastAPI deployment allows easy integration with other systems and provides a user-friendly interface for conducting diabetes risk assessments. Users can interact with our model effortlessly, making informed healthcare decisions based on our predictions.
![FastAPI Interface](https://github.com/LohithUnnam/Diabetes-Prediction-MLOps/blob/main/assets/fastapi_img.png)
## Model Monitoring with Streamlit and Evidently
In our project, we've developed a Streamlit application that incorporates Evidently for monitoring the performance of our diabetes prediction model. This interactive application empowers healthcare professionals and individuals to gain insights into the model's behavior, assess its accuracy, and make informed decisions.

### Key Components:

Streamlit Application: We've created a Streamlit web application, providing a user-friendly interface for monitoring our diabetes prediction model.

Data Loading: Our application loads pre-processed data from a CSV file, preparing it for analysis.

Model Selection: Users can choose from a set of models, including Random Forest, Logistic Regression, and Support Vector Machine (SVM).

Model Training: The selected model is trained on a subset of the available data, ensuring that the model aligns with the current dataset.

Probabilistic Predictions: Instead of making binary predictions, the application produces probabilistic predictions using the predict_proba method. These probabilities indicate the likelihood of an individual having diabetes.

Column Mapping: We configure Evidently's ColumnMapping to define the target column, the prediction column (probability), and the feature columns. This setup allows for precise monitoring.

Classification Report: We leverage Evidently's classification metrics preset to generate a detailed classification report. This report provides a comprehensive assessment of the model's performance, including accuracy, precision, recall, and more.

Interactive User Interface: Users can view the classification report within the Streamlit application, providing a dynamic and engaging experience for model monitoring.

#### User Experience:

Users can choose a model type (Random Forest, Logistic Regression, or SVM) in the application's sidebar.

The selected model is trained on a sample dataset to produce probabilistic predictions.

Evidently's classification report is generated, displaying key model performance metrics.

Users can interact with the report to gain insights into the model's strengths and weaknesses, ultimately making data-driven decisions.

This Streamlit application, combined with Evidently's monitoring capabilities, enhances the transparency and accountability of our diabetes prediction model. Users can effectively assess the model's accuracy and reliability, supporting proactive healthcare and well-being.

![This is how our streamlit application will look like](https://github.com/LohithUnnam/Diabetes-Prediction-MLOps/blob/main/assets/stream_img.png)

This comprehensive MLOps project focuses on Diabetes Prediction and utilizes a range of tools, including Prefect, MLflow, FastAPI, and Streamlit.
