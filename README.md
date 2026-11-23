
# 🩺 Breast Cancer Prediction System

A Machine Learning-based web application that predicts whether a breast tumor is **Malignant** or **Benign** using diagnostic features.

Built with Python, Scikit-learn, and Streamlit.

---

## 📌 Project Overview

The Breast Cancer Prediction System is an academic Machine Learning project developed to explore and compare different classification algorithms for breast cancer prediction.

The system uses three Machine Learning models:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest

Users can make predictions, compare model performance, and explore evaluation results through an interactive dashboard.

## 📊 Dataset

**Wisconsin Breast Cancer Diagnostic Dataset**

| Description | Value |
|---|---:|
| Total Samples | 569 |
| Total Features | 30 |
| Training Samples | 455 |
| Testing Samples | 114 |

## 🤖 Machine Learning Models

| Model | Accuracy |
|---|---:|
| Logistic Regression | 98.25% |
| K-Nearest Neighbors | 95.61% |
| Random Forest | 95.61% |

## ✨ Application Features

- Interactive Dashboard
- Breast Cancer Prediction
- Prediction Confidence
- Model Comparison
- Model Evaluation
- Confusion Matrix
- Classification Report
- Prediction History
- Clear Prediction History

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook

## 📂 Project Structure

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

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone git@github.com:nayeem-khan9/breast-cancer-prediction.git
```

**2. Navigate to the project directory**

```bash
cd breast-cancer-prediction
```

**3. Install dependencies**

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn joblib
```

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📓 Jupyter Notebook

The `breast_cancer_analysis.ipynb` notebook contains the complete Machine Learning workflow, including data preprocessing, model training, and evaluation.

## ⚠️ Disclaimer

This application is developed for educational and academic purposes only. It is not a medical diagnostic tool and should not replace professional medical advice.

---

**Developed by [Nayeem Khan](https://github.com/nayeem-khan9)**
