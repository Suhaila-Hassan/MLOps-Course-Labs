# Bank Customer Churn Prediction
This project is designed to predict customer churn for a bank using various machine learning models and includes full ML lifecycle management and deployment. The solution is split into two main parts:
1. Running experiments, training different models and evaluating them inside train.py and MLflow
2. Model deployment using FastAPI inside main.py
3. Containerization using Docker

# Dataset
The dataset used is Churn_Modelling.csv, which contains information on bank customers such as credit score, geography, age, balance, and whether they exited the bank.

Target column: Exited
- 0: Stayed
- 1: Churned

# Projects Phases
1. Loading Dataset
2. Data preprocessing
3. Model training:
   - Logistic Regression
   - Random Forest (Best Performing Model)
   - XGBoost
4. Model evaluation using Accuracy, Precision, Recall, F1-score and Confusion Matrix
5. Experiment tracking using MLflow
6. FastAPI-based REST API for inference
7. Docker support for deployment

# Model Training (train.py)
train.py is responsible for:
- Reading and preprocessing the dataset
- Handling class imbalance through downsampling
- Splitting the data into training and testing sets
- Training Logistic Regression, Random Forest, and XGBoost models
- Evaluating models and logging metrics with MLflow
- Saving trained models and transformers to disk

## Key Functions
- rebalance(data): Downsamples the majority class
- preprocess(df): Prepares features and saves the column transformer
- train(model_class, params, X_train, y_train): Trains a given model
- evaluate(y_true, y_pred, model_name): Logs performance metrics and plots
- run_experiment(...): Combines training, evaluation, and MLflow logging

## Outputs
- Saved model files in models/
- Confusion matrix plots in plots/
- MLflow logs (metrics, artifacts, model parameters)

# API Inference (main.py)
main.py launches a RESTful API using FastAPI to serve the churn prediction model.
## Endpoints
- GET /: Returns a welcome message
- GET /health: Health check endpoint
- POST /predict: Accepts customer data and returns churn prediction

# Docker Deployment
A Python-based Docker container is provided to run the FastAPI app.

