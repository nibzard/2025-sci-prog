# Medical Insurance Cost Prediction

---

## Introduction

Health insurance companies estimate medical costs for their clients based on demographic and lifestyle characteristics such as age, BMI, smoking status, and number of children.  
Accurate cost estimation is important for pricing strategies, risk assessment, and long-term sustainability of insurance systems.

Machine learning techniques allow us to analyze historical insurance data, identify key cost drivers, and build predictive models that estimate expected medical expenses for individuals.

The goal of this project is to build a **regression-based machine learning pipeline** that predicts **medical insurance charges** using personal and health-related attributes.

---

## Problem Statement

Medical insurance costs vary significantly between individuals and are influenced by multiple interacting factors.  
Simple linear assumptions often fail to capture non-linear effects such as the combined impact of smoking and BMI.

This project addresses the challenge of:
- understanding **which factors most strongly influence insurance costs**, and
- building a **reproducible predictive system** that estimates insurance charges with good accuracy.

---

## Hypothesis & Objectives

**Hypothesis:**  
Medical insurance costs can be predicted with reasonable accuracy using demographic and lifestyle variables, with **smoking status, BMI, and age** expected to be the most significant predictors.

**Objectives:**
- Build an end-to-end machine learning pipeline
- Compare multiple regression models
- Evaluate model performance using standard regression metrics
- Interpret model results and feature importance
- Ensure the project is reproducible and runnable from the command line

---

## Dataset

The dataset used is the **Medical Insurance Cost Dataset** from Kaggle:  
https://www.kaggle.com/datasets/mirichoi0218/insurance/data

- **Rows:** 1338  
- **Target variable:** `charges`

| Feature | Description |
|-------|-------------|
| age | Age of the individual |
| sex | Gender (male/female) |
| bmi | Body Mass Index |
| children | Number of dependent children |
| smoker | Smoking status |
| region | US residential region |
| charges | Medical insurance cost |

The dataset contains no missing values and is well-suited for regression analysis.

---

## Success Criteria

The project is considered successful if:
- Models achieve a **high R² score** and **low RMSE**
- Results confirm or reject the initial hypothesis
- The pipeline can be reproduced locally using simple commands
- Feature importance provides meaningful real-world insights

---

## Architecture & Implementation

The project follows a **modular Python architecture** and runs entirely from the command line.

### High-Level Architecture

```mermaid
flowchart LR
    A["insurance.csv"] --> B["Data Loading"]
    B --> C["Exploratory Data Analysis"]
    C --> D["Feature Engineering"]
    D --> E["Model Training"]
    E --> F["Model Evaluation"]
    F --> G["Saved Model & Metrics"]
```

### Project Structure

```text
mlovric-insurancecost/
├── README.md                     # Project documentation
├── run.py                        # Main CLI entry point (eda/train/eval)
├── requirements.txt              # Python dependencies
├── pytest.ini                    # Testing configuration
│
├── data/
│   └── insurance.csv             # Kaggle insurance dataset
│
├── src/
│   ├── main.py                   # Routes CLI commands
│   ├── data_loader.py            # Loads and inspects dataset
│   ├── preprocessing.py          # Encoding, scaling, train-test split
│   ├── features.py               # Feature engineering logic
│   ├── eda.py                    # Exploratory data analysis (plots)
│   ├── models.py                 # Model definitions
│   ├── train.py                  # Training pipeline
│   ├── evaluate.py               # Metrics and model comparison
│   └── interpretation.py         # Feature importance and insights
│
├── reports/
│   ├── figures/                  # Saved plots (EDA & evaluation)
│   └── metrics/
│       └── model_comparison.csv  # MAE, MSE, RMSE, R² results
│
├── models/
│   └── best_model.joblib         # Best trained model
│
└── tests/
    └── test_basic.py             # Basic unit tests
```

## Technical Decisions & Rationale
- Python & scikit-learn were chosen for accessibility and reproducibility
- Tree-based ensemble models were included to capture non-linear relationships
- Interaction features were added based on domain knowledge
- Command-line execution was used instead of notebooks to follow software engineering best practices

Trade-offs considered:
- Linear models are interpretable but limited
- Ensemble models offer higher accuracy but lower interpretability


## Reproducibility / How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python run.py eda
python run.py train
python run.py eval
```


## Results & Conclusions

All models achieved strong predictive performance, with **R² values above 0.86**.  
The **Gradient Boosting Regressor** achieved the best overall performance, with the **lowest RMSE** and **highest R² score**.

Feature importance analysis showed:
- **Smoking × BMI** is the strongest predictor of insurance costs
- **Age** has a significant independent effect
- **Interaction terms** substantially improve predictive accuracy

**Conclusion:**  
The hypothesis was **confirmed**. Medical insurance costs can be predicted with good accuracy using demographic and lifestyle variables, with smoking status being the most influential factor.

---

## Limitations

- No medical history or chronic condition data
- Limited dataset size
- Regional categories are broad

---

## Next Steps & Opportunities

- Hyperparameter tuning
- Adding additional medical or lifestyle features
- Deploying the model as a REST API
- Bias and fairness analysis in insurance pricing
- Scaling the pipeline to larger healthcare datasets

