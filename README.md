# Student Depression Analysis & Prediction

This project analyzes a dataset on student mental health to identify key factors correlated with depression and build a predictive machine learning model.

## 📊 Project Overview

- **Dataset:** [Student Depression Dataset](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset)
- **Goal:** Understand which academic and lifestyle factors correlate with depression, and build a classifier to predict depression risk.
- **Final Model:** Random Forest Classifier
- **Accuracy:** ~84%

## Techniques Used

- Data cleaning and preprocessing (handling missing values, encoding, filtering)
- Exploratory Data Analysis (EDA) using boxplots, group-by stats, and correlation heatmaps
- Feature engineering and encoding
- Machine Learning with Scikit-learn
- Model evaluation with precision, recall, F1-score, and confusion matrix

## Key Insights - Selective Data Points

- **Sleep Duration**: Shorter sleep is generally associated with higher depression, up to a certain point, at which the same effects arent seen.
- **Academic Pressure**: Strong positive correlation with depression.
- **CGPA**: No clear linear trend with depression, seemingly noise.
- **Degree Type**: 'Class 12' students showed significantly higher depression risk than university students.

## ML Model

- Model: Random Forest Classifier  
- Evaluation:  
  - Precision (Depressed): 0.86  
  - Recall (Depressed): 0.87  
  - F1-score (Depressed): 0.86  
  - Overall Accuracy: 84%

## Files

- `student_depression_analysis.ipynb` – Full notebook with data cleaning, EDA, and modeling
- `student_depression.csv` – Dataset used (or link to Kaggle if not included)

## How to Run

1. Clone this repository  
2. Open the Jupyter notebook  
3. Run cells in order (requires pandas, seaborn, scikit-learn, matplotlib)

