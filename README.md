# Climate Temperature Forecasting Using Machine Learning  
**OLS, Random Forest, Gradient Boosting, Diagnostics, Bootstrap Intervals & Ablation Study (1850–2024)**

This project develops a complete and reproducible **machine learning pipeline** to forecast global temperature anomalies using engineered time-series features and classical ML models.  
Using data from **1850–2024**, it achieves strong predictive performance and generates calibrated 10-year climate forecasts (2025–2034).

---

## 👥 Team Members
- **Shreyash Naidu Mamidi**
- **Aneesh Reddy Koppurapu**
- **Jyothsna Anne**
- **Chamundeshwari Batti**
- **Abhinav Sai Ratan Attemla**
- **Ritvik Subramanyam tolety **
- **Gayathri Thota**
- 
**Institution:** Virginia Commonwealth University  
**Course:** CMSC 535 Introduction to Data Science
**Instructor:** Dr. Thomas W Gyeera  
**Semester:** Fall 2025

---

## 📋 Project Overview
This project forecasts future global temperature anomalies using:

- Multi-year **lag features**
- **Rolling-window** climate signals
- **Trend components**
- Classical ML models (OLS, RF, GBR)
- **Expanding-window evaluation** (time-series safe)
- **Residual diagnostics**
- **Bootstrap prediction intervals**
- **Ablation study** to understand feature importance

The pipeline is designed to be transparent, reproducible, and aligned with scientific forecasting methodologies.

---

## 🌍 Problem Statement  
Global temperatures have risen dramatically, and **accurate long-term forecasting** is essential for climate policy, environmental planning, and scientific analysis.

This project creates a robust forecasting system using historical temperature anomaly data to predict trends for the next decade (2025–2034).

---

## 🚀 Features  
- ✅ 175+ years of climate data (1850–2024)  
- ✅ Lag features (1–5 year history)  
- ✅ Rolling averages (5-year & 11-year climate signals)  
- ✅ Linear and nonlinear ML models  
- ✅ Expanding-window time-series evaluation  
- ✅ Full residual diagnostics  
- ✅ 10-year forecast using Random Forest  
- ✅ Bootstrap uncertainty intervals  
- ✅ Full ablation study on feature groups  
- ✅ Results exported as CSV + plotted figures  

---

## 📊 Dataset  
The dataset includes annual global temperature anomalies from 1850 to 2024.

### Summary
- **Years:** 1850–2024  
- **Samples:** 175  
- **Type:** Regression (continuous climate anomaly prediction)

Dataset used: *Annual global temperature anomaly values (°C)*.

---

## 🧠 Feature Engineering  

### Engineered Inputs
| Feature | Description |
|---------|-------------|
| lag_1 … lag_5 | 1–5 year historical anomalies |
| roll_5 | 5-year moving average |
| roll_11 | 11-year moving average |
| year_idx | Trend index |

These features capture **short-term variability**, **medium-term smoothing**, and **long-term warming trend**.

---

## 🤖 Models Implemented  

### 1️⃣ Ordinary Least Squares (OLS)
- Baseline linear regression  
- Excellent for global warming trend detection  

### 2️⃣ Random Forest (RF)
- Captures nonlinear relationships  
- Good for short-term climate variability  

### 3️⃣ Gradient Boosting Regressor (GBR)
- Gradient-boosted trees  
- Captures subtle nonlinear acceleration patterns  

---

## 📉 Evaluation — Expanding Window  
A time-series safe evaluation method:
Train on 1850–1900 → test 1901
Train on 1850–1901 → test 1902

This prevents **data leakage** and simulates real-world forecasting.

---

## 🏆 Model Performance Summary

| Model | RMSE | MAE | R² |
|--------|--------|--------|-------|
| **OLS** | **0.1087** | **0.0896** | **0.9198** |
| **RF** | 0.1181 | 0.0977 | 0.9051 |
| **GBR** | 0.1218 | 0.1008 | 0.8991 |

**OLS performs best** → Indicates global warming follows a *mostly linear long-term trend*.

---

## 🔍 Residual Diagnostics (OLS)
| Metric | Value |
|--------|--------|
| Mean residual | 0.0311 |
| Std residual | 0.1041 |
| Durbin–Watson | **1.9052** |

DW ≈ 2 → *No autocorrelation* → Model is statistically sound.

---

## 🔮 10-Year Forecast (2025–2034)

Using Random Forest:

| Year | Forecast (°C Anomaly) |
|------|------------------------|
| 2025 | 1.129  
| 2026 | 1.049  
| 2027 | 1.056  
| 2028 | 1.093  
| 2029 | 1.103  
| 2030–2034 | 1.10–1.17 |

Forecast shows **persistent warming beyond 1.0°C**.

---

## 🎯 Bootstrap Prediction Interval Example
For year **2034**:

- **Lower:** 0.86  
- **Median:** 1.09  
- **Upper:** 1.17  

Tight uncertainty band → high confidence in warming trend.

---

## 🧪 Ablation Study (Feature Group Importance)

| Feature Group | Best Model | R² |
|----------------|------------|------|
| Lags Only | OLS | 0.917 |
| Rolling Only | OLS | 0.906 |
| Trend Only | RF | 0.916 |
| Lags + Rolling | OLS | 0.920 |
| **Full** | OLS | **0.920** |

### Conclusions
- **Lags** are the strongest individual predictor  
- **Trend** matters more for nonlinear models  
- **Rolling averages** help stabilize predictions  
- **Full feature set** performs best  

---

## 📁 Project Structure
```
climate-temp-forecast/
│── main.py
│── requirements.txt
│── README.md
│── .gitignore
│
├── data/
│ └── annual.csv
│
├── results/
│ ├── metrics/
│ │ ├── main_results.csv
│ │ ├── ablation_results.csv
│ │ ├── calibration_intervals.csv
│ │ └── forecast_next_10_years.csv
│ │
│ └── figures/
│ ├── acf_ols.png
│ ├── residuals_ols.png
│ ├── feature_importance_rf.png
│ ├── forecast_rf.png
│ └── (more figures)
│
└── src/
├── data_loader.py
├── feature_engineering.py
├── models.py
├── evaluation.py
├── diagnostics.py
├── forecast.py
├── calibration.py
└── ablation.py
```
---

## 🧪 Installation & Running the Pipeline

Install dependencies:
```bash
pip install -r requirements.txt
```
Run full pipeline:
```bash
python main.py
```
Outputs are saved to:
```bash
results/metrics/
results/figures/
```
## 📚 References
1. NASA GISTEMP Analysis
2. Scikit-learn documentation
3. Statsmodels time-series analysis
4. Climate literature on global anomaly modeling
##🎓 Acknowledgments
-Virginia Commonwealth University
-CMSC 630 – Image Analysis
-Dr. Wei-Bang Chen
-NASA GISTEMP
-Scikit-learn developers


