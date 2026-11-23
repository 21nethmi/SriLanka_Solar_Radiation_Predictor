# Sri Lankan Solar Radiation Prediction Using Ensemble Learning

## 1. Project Overview

This project develops a machine learning-based system to estimate daily solar radiation in selected locations in Sri Lanka using weather and geographical information.

The system combines weather data and solar radiation data to train and evaluate different ensemble learning approaches. The trained models are integrated into an interactive Streamlit dashboard that allows users to explore model performance and estimate solar radiation based on supplied weather conditions.

Solar radiation estimation can support preliminary solar energy planning by helping users understand the solar potential of different locations. It can also provide useful information for planning solar panel installations and solar power station development.

**Note:** This project estimates solar radiation using supplied same-day weather conditions. It is not a future-weather forecasting system.

## 2. Project Objectives

- Analyse and combine Sri Lankan weather and solar datasets.
- Develop regression models using ensemble learning techniques.
- Implement and compare Bagging, Boosting, Stacking and Hybridization.
- Evaluate model performance using MAE, RMSE and R².
- Interpret model behaviour using SHAP explainability.
- Develop an interactive dashboard for solar radiation estimation and model comparison.

## 3. Datasets

The project uses two datasets from the Lanka Data Tracker repository.

- `weather.csv` – contains daily weather information such as maximum and minimum temperature, precipitation, maximum wind speed and mean humidity.
- `solar.csv` – contains daily solar information, including solar radiation, sunshine hours, daylight hours and UV index.

**Data source:**  
https://github.com/dhananjaya-hbc/lanka-data-tracker

The datasets are merged using the date and city fields.

## 4. Machine Learning Models

Four approaches are implemented:

### 4.1 Bagging – Random Forest Regressor

Uses multiple decision trees trained on different samples to produce a combined prediction.

### 4.2 Boosting – XGBoost Regressor

Builds decision trees sequentially, with each stage learning to improve the errors made by previous stages.

### 4.3 Stacking – Random Forest + XGBoost + ANN

Combines predictions from Random Forest, XGBoost and an Artificial Neural Network (ANN). Ridge Regression is used as the meta-learner to generate the final prediction.

### 4.4 Hybridization – Random Forest + ANN

Combines Random Forest and ANN predictions using a weighted average, with an initial weight of 50% for each model.

## 5. Model Performance

The models were evaluated using a chronological training and testing split.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 0.4878 | 0.6386 | 0.7233 |
| XGBoost | 0.4846 | 0.6313 | 0.7296 |
| Stacking | 0.4354 | 0.5947 | 0.7601 |
| RF + ANN Hybrid | 0.4581 | 0.6029 | 0.7534 |

Stacking achieved the best performance among the four evaluated approaches on the test dataset.

## 6. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib and Seaborn
- Streamlit
- Joblib

## 7. Project Structure

```text
SriLanka_Solar_Ensemble/
│
├── SriLanka_Solar.ipynb
├── weather.csv
├── solar.csv
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── stacking_model.pkl
│   └── hybrid_model.pkl
│
└── results/
    ├── model_metrics.csv
    └── test_predictions.csv
```

## 8. Running the Dashboard

### Step 1: Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit application

```bash
streamlit run app.py
```

The dashboard will normally be available at:

```text
http://localhost:8501
```

## 9. Dashboard Features

The dashboard is planned to provide:

- Project overview and dataset information.
- Solar radiation estimation using weather inputs.
- Comparison of all four ensemble learning approaches.
- Evaluation metric visualizations.
- Actual versus predicted solar radiation analysis.
- Model insights and SHAP-based explainability.

