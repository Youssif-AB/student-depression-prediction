# Student Depression Analysis & Prediction

This project explores academic, lifestyle, and demographic factors associated with the `Depression` label and trains a classifier using the workflow developed in the notebook.

## Dataset

The included `student_depression_dataset.csv` is the [Student Depression Dataset on Kaggle](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset). It contains self-reported or survey-style student attributes including academic pressure, study satisfaction, sleep duration, dietary habits, financial stress, family mental-illness history, and prior suicidal thoughts. The binary `Depression` column is the prediction target.

The dataset is observational and its provenance, sampling process, label construction, and representativeness are not established in this repository. The model can reproduce biases and data-quality problems in the dataset. Its held-out performance is specific to one split of this data and has not been externally validated. Because it uses sensitive mental-health information, including prior suicidal thoughts, it must not be treated as a diagnosis, used for clinical decisions, or deployed to make decisions about individuals.

## Existing Methodology

The notebook drops `id`, `City`, and `Work Pressure`; retains student records; removes unsupported categorical values; maps binary fields; ordinal-encodes dietary habits and sleep duration; and one-hot-encodes degree with the first level dropped. It uses an 80/20 stratified train/test split with random state 42 and trains one candidate: `RandomForestClassifier(random_state=42)` with otherwise default parameters.

The existing evaluation reports mean absolute error, a confusion matrix, and classification precision, recall, F1, and accuracy. No additional candidate or tuning methodology is introduced by the tracked workflow.

## MLflow Training

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the tracked workflow:

```bash
python train.py
```

`MLFLOW_TRACKING_URI` defaults to `http://localhost:5000` and can be overridden:

```powershell
$env:MLFLOW_TRACKING_URI = "http://another-mlflow-host:5000"
python train.py
```

The script logs the real estimator parameters and held-out metrics to the `student-wellness-prediction` experiment, logs the trained scikit-learn model artifact, and registers it as `StudentWellnessPrediction`. It is independent of ModelControl source code and communicates only through the standard MLflow tracking API.

## Files

- `student-depression-prediction.ipynb`: exploratory analysis and original modeling workflow
- `student_depression_dataset.csv`: dataset used by the notebook and training script
- `train.py`: reproducible MLflow-tracked version of the existing training/evaluation workflow
- `requirements.txt`: runtime dependencies
