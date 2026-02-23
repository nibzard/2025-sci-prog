# Sales Forecasting Machine Learning Project

This project demonstrates the application of machine learning techniques to time series sales forecasting. The goal is to predict short-term daily retail sales using historical data, with a focus on correct time series methodology and practical business relevance.

## 🎯 Overview

The project develops a sales forecasting pipeline capable of predicting daily sales for the next 14–30 days. It applies feature engineering, baseline and advanced forecasting models, and standard evaluation metrics to compare model performance.

The emphasis is placed on:

- proper handling of time-dependent data,
- avoiding data leakage,
- interpretability of results,
- applicability of forecasts in a business context

## 💼 Business Context

Accurate sales forecasting is essential for inventory management in retail. Poor forecasts can result in overstocking, increased costs, or stockouts and lost revenue.

This project shows how data-driven forecasting models can support inventory planning and improve decision-making by estimating future demand more reliably than simple heuristic approaches.

## 📊 Dataset

The dataset used is Corporación Favorita Store Sales from Kaggle, containing daily sales data from an Ecuadorian grocery retailer.

For clarity and manageability, the analysis focuses on:

- one store (Store 44),
- one product family (GROCERY I),
- daily sales data over multiple years.

Key variables include the sale date, store identifier, product family, number of units sold, and promotion information.

- [Dataset Link](https://www.kaggle.com/c/store-sales-time-series-forecasting)

## 📁 Project Structure

sales-forecasting/
├── data/ # Raw and processed datasets
├── src/ # Data processing and modeling code
├── notebooks/ # Jupyter notebooks for analysis and modeling
├── figures/ # Generated plots and results
├── requirements.txt
└── README.md

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd sales-forecasting
   ```

2. **Create virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset**:
   - Download `train.csv` from [Kaggle Competition](https://www.kaggle.com/c/store-sales-time-series-forecasting/data)
   - Place in `data/` folder

---

## ▶️ How to Run

### Quick Start

Run notebooks in sequential order:

```bash
jupyter notebook
```

Then execute:

1. `01_exploratory_analysis.ipynb` - EDA and scope selection
2. `02_preprocessing_feature_engineering.ipynb` - Feature engineering
3. `03_baseline_models.ipynb` - Baseline models
4. `04_main_models.ipynb` - Advanced models (SARIMA, RF, GB)
5. `05_evaluation_and_comparison.ipynb` - Comprehensive evaluation
6. `06_business_use_case.ipynb` - Inventory planning application

#### 4. **Evaluation Metrics**

-MAE,RMSE

## 🤖 Models

### Baseline Models

1. **Naive Forecaster**
   - Method: Tomorrow = Today
   - Purpose: Simplest possible baseline

2. **Seasonal Naive**
   - Method: Tomorrow = Same day last week
   - Purpose: Captures weekly seasonality

3. **Moving Average (7-day)**
   - Method: Tomorrow = Average of last 7 days
   - Purpose: Smoothed baseline

4. **Linear Regression**
   - Method: Linear model with all features
   - Purpose: Simple ML baseline

### Advanced Models

5. **SARIMA (Seasonal ARIMA)**
   - Order: (1,1,1)×(1,1,1,7)
   - Statistical time series model
   - Captures trend + weekly seasonality

6. **SARIMAX**
   - SARIMA + exogenous variables (promotions)
   - Tests impact of external factors

7. **RandomForest**
   - Ensemble of 200 decision trees
   - Captures non-linear patterns
   - Robust to outliers

8. **GradientBoosting**
   - Sequential boosting algorithm
   - Often best performance
   - Feature importance analysis

---

All visualizations are saved to `figures/` directory:
