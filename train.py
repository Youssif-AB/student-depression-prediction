"""Train the notebook's Random Forest workflow and track it with MLflow."""

import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "student_depression_dataset.csv"
EXPERIMENT_NAME = "student-wellness-prediction"
REGISTERED_MODEL_NAME = "StudentWellnessPrediction"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Apply the notebook's existing feature selection and encoding exactly."""
    prepared = data.drop(columns=["id", "City", "Work Pressure"]).copy()
    prepared = prepared[prepared["Gender"] != "Others"].copy()
    ordinal_encoder = OrdinalEncoder()
    prepared[["Dietary Habits"]] = ordinal_encoder.fit_transform(prepared[["Dietary Habits"]])
    prepared = prepared[prepared["Profession"] == "Student"].copy()
    prepared["Have you ever had suicidal thoughts ?"] = prepared["Have you ever had suicidal thoughts ?"].map({"Yes": 1, "No": 0})
    prepared["Gender"] = prepared["Gender"].map({"Male": 1, "Female": 0})
    prepared["Family History of Mental Illness"] = prepared["Family History of Mental Illness"].map({"Yes": 1, "No": 0})
    prepared = prepared[prepared["Financial Stress"] != "?"].copy()
    prepared = prepared.drop(columns=["Profession"])
    prepared[["Sleep Duration"]] = ordinal_encoder.fit_transform(prepared[["Sleep Duration"]])

    one_hot_encoder = OneHotEncoder(sparse_output=False, drop="first")
    encoded_degree = one_hot_encoder.fit_transform(prepared[["Degree"]])
    degree_frame = pd.DataFrame(encoded_degree, columns=one_hot_encoder.get_feature_names_out(["Degree"]))
    prepared.index = degree_frame.index
    prepared = pd.concat([prepared, degree_frame], axis=1)
    prepared = prepared.rename(columns={"Have you ever had suicidal thoughts ?": "Suicidal Thoughts"}).drop(columns=["Degree"])
    return prepared.drop(columns=["Depression"]), prepared["Depression"]


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    features, target = prepare_data(data)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "mean_absolute_error": mean_absolute_error(y_test, predictions),
        "precision_class_0": report["0"]["precision"],
        "recall_class_0": report["0"]["recall"],
        "f1_class_0": report["0"]["f1-score"],
        "precision_class_1": report["1"]["precision"],
        "recall_class_1": report["1"]["recall"],
        "f1_class_1": report["1"]["f1-score"],
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
        "precision_weighted": report["weighted avg"]["precision"],
        "recall_weighted": report["weighted avg"]["recall"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        "true_negative": int(matrix[0, 0]),
        "false_positive": int(matrix[0, 1]),
        "false_negative": int(matrix[1, 0]),
        "true_positive": int(matrix[1, 1]),
    }

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="random-forest") as run:
        mlflow.log_params({
            **model.get_params(),
            "dataset_rows": len(data),
            "prepared_rows": len(features),
            "feature_count": features.shape[1],
            "test_size": 0.2,
            "split_random_state": 42,
            "split_stratified": True,
            "degree_encoding": "one-hot-drop-first",
            "ordinal_encoded_features": "Dietary Habits,Sleep Duration",
        })
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "candidate": "RandomForestClassifier",
            "selection": "champion",
            "source_workflow": "student-depression-prediction.ipynb",
        })
        signature = infer_signature(x_train, model.predict(x_train))
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=x_test.head(5),
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        print(f"tracking_uri={TRACKING_URI}")
        print(f"experiment={EXPERIMENT_NAME}")
        print(f"run_id={run.info.run_id}")
        print(f"model_uri={model_info.model_uri}")
        for name, value in metrics.items():
            print(f"{name}={value}")


if __name__ == "__main__":
    main()
