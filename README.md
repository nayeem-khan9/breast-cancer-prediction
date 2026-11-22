
# Breast Cancer Prediction System

## Project Overview
The Breast Cancer Prediction System is a Machine Learning-based web application that predicts whether a breast tumor is Malignant or Benign using diagnostic features.

The application uses three Machine Learning algorithms:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest

## Dataset
- Dataset: Wisconsin Breast Cancer Diagnostic Dataset
- Total Samples: 569
- Total Features: 30
- Training Samples: 455
- Testing Samples: 114

## Machine Learning Workflow
1. Data Loading and Understanding
2. Data Preprocessing
3. Feature and Target Separation
4. Train-Test Split
5. Model Training
6. Model Evaluation
7. Model Saving
8. Model Integration with Streamlit

## Model Performance

| Model | Accuracy |
|---|---:|
| Logistic Regression | 98.25% |
| KNN | 95.61% |
| Random Forest | 95.61% |

## Application Features
- Interactive Dashboard
- Breast Cancer Prediction
- Prediction Confidence
- Model Comparison
- Model Evaluation
- Confusion Matrix
- Classification Report
- Prediction History
- Clear Prediction History

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook

## Project Structure

```text
breast-cancer-prediction/
│
├── app.py
├── README.md
├── breast_cancer_analysis.ipynb
│
├── data/
│   ├── breast_cancer.csv
│   └── test_data.csv
│
└── models/
    ├── logistic_model.pkl
    ├── knn_model.pkl
    └── random_forest_model.pkl
```

## Installation

Install the required libraries:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn joblib
```

## Run the Application

Open the terminal inside the project directory and run:

```bash
streamlit run app.py
```

## Disclaimer
This application is developed for educational and academic purposes only. It is not a medical diagnostic tool and should not replace professional medical advice.
