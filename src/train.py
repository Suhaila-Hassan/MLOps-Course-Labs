"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
os.environ["LOGNAME"] = "Suhaila"
import mlflow
import mlflow.sklearn

RANDOM_STATE = 42
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Bank_Customer_Churn_Experiments")


def rebalance(data):
    """
    Resample data to keep balance between target classes.

    Args:
        data (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame: Balanced DataFrame
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj, churn_min = churn_0, churn_1
    else:
        churn_maj, churn_min = churn_1, churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=RANDOM_STATE
    )
    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): Raw input DataFrame

    Returns:
        ColumnTransformer: Transformer for preprocessing
        np.array: X_train
        np.array: X_test
        pd.Series: y_train
        pd.Series: y_test
    """
    filter_feat = [
        "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )

    col_transf = make_column_transformer(
        (StandardScaler(), num_cols),
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_test = col_transf.transform(X_test)

    return col_transf, X_train, X_test, y_train, y_test


def train(model_class, params, X_train, y_train):
    """
    Train a model with given parameters and training data.

    Args:
        model_class: The model class (e.g., LogisticRegression)
        params (dict): Parameters to initialize the model
        X_train: Training features
        y_train: Training targets

    Returns:
        Trained model instance
    """
    model = model_class(**params)
    model.fit(X_train, y_train)
    return model


def evaluate(y_true, y_pred, model_name):
    """
    Evaluate the model and log metrics and confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        model_name: Name of the model
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    conf_mat = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    disp.plot()
    os.makedirs("plots", exist_ok=True)
    plot_path = f"plots/conf_matrix_{model_name}.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    plt.close()


def run_experiment(model, model_name, X_train, X_test, y_train, y_test, params={}):
    """
    Run training and evaluation experiment with MLflow tracking.

    Args:
        model: Untrained model instance
        model_name: Name to identify model in logs
        X_train: Training features
        X_test: Test features
        y_train: Training targets
        y_test: Test targets
        params: Model hyperparameters
    """
    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("model_type", model_name)
        mlflow.log_params(params)

        trained_model = train(type(model), params, X_train, y_train)
        y_pred = trained_model.predict(X_test)

        evaluate(y_test, y_pred, model_name)

        os.makedirs("models", exist_ok=True)
        model_path = f"models/{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(trained_model, f)

        mlflow.sklearn.log_model(trained_model, model_name)


def main():
    df = pd.read_csv("dataset/Churn_Modelling.csv")
    _, X_train, X_test, y_train, y_test = preprocess(df)

    # Logistic Regression
    run_experiment(
        LogisticRegression(),
        "LogisticRegression",
        X_train,
        X_test,
        y_train,
        y_test,
        {"max_iter": 1000, "random_state": RANDOM_STATE}
    )

    # Random Forest
    run_experiment(
        RandomForestClassifier(),
        "RandomForest",
        X_train,
        X_test,
        y_train,
        y_test,
        {"n_estimators": 100, "max_depth": 10, "random_state": RANDOM_STATE}
    )

    # XGBoost
    run_experiment(
        XGBClassifier(),
        "XGBoost",
        X_train,
        X_test,
        y_train,
        y_test,
        {"use_label_encoder": False, "eval_metric": "logloss", "random_state": RANDOM_STATE}
    )


if __name__ == "__main__":
    main()
