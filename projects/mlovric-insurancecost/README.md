# Medical Insurance Cost Prediction

---

## Introduction

Health insurance companies estimate medical costs for their clients based on demographic and lifestyle characteristics such as age, BMI, smoking status, and number of children.  
Machine learning techniques can help identify key cost drivers and make accurate predictions of expected insurance expenses.

The goal of this project is to build a **regression-based machine learning model** that predicts **medical insurance charges** using personal and health-related attributes.

Such a model can support insurance companies in pricing strategies, risk assessment, and understanding factors that most influence healthcare spending.

---

## Hypothesis

Medical insurance costs can be predicted with reasonable accuracy using demographic and lifestyle variables, with smoking status, BMI, and age expected to be the most significant predictors.

---

## Dataset

The dataset used in this project is the **Medical Insurance Cost Dataset**, available on Kaggle:  
**https://www.kaggle.com/datasets/mirichoi0218/insurance/data**

It contains **1338 rows** and the following attributes:

| Attribute | Description |
|----------|-------------|
| **age** | Age of the individual |
| **sex** | Gender (male/female) |
| **bmi** | Body Mass Index |
| **children** | Number of dependent children |
| **smoker** | Smoker status (yes/no) |
| **region** | Residential US region (northwest, southwest, southeast, northeast) |
| **charges** | Medical insurance cost (target variable) |

The dataset is adapted from materials in the book *Machine Learning with R* by Brett Lantz.

---

## Methodology

### 1. Data Preparation
- Import and inspect dataset  
- Check for missing values  
- Encode categorical variables (Label Encoding / One-Hot Encoding)  
- Scale numerical features if needed  

### 2. Exploratory Data Analysis (EDA)
- Distribution plots for age, BMI, charges  
- Boxplots to examine the effect of smoking, sex, and region  
- Correlation heatmap  
- Analysis of key drivers of insurance cost  

### 3. Feature Engineering
- Create additional meaningful features, such as:  
  - `bmi_category` (underweight / normal / overweight / obese)  
  - `is_smoker` as binary 0/1  
  - Interaction terms (e.g., smoker × bmi, age × bmi)  
- Normalize continuous features for certain models  

### 4. Modeling
- Split the data into training (80%) and testing (20%) sets  
- Train and compare multiple regression models:  
  - Linear Regression  
  - Ridge Regression  
  - Lasso Regression  
  - Random Forest Regressor  
  - Gradient Boosting Regressor  

### 5. Evaluation
- Use the following metrics:  
  - MAE (Mean Absolute Error)  
  - MSE (Mean Squared Error)  
  - RMSE (Root Mean Squared Error)  
  - R² Score  
- Visual comparison of model performance  

### 6. Interpretation
- Identify most important features  
- Discuss how smoking, BMI, and age influence insurance charges  
- Evaluate model limitations and potential improvements  
- Provide practical insights for real-world health insurance pricing  

---

## Project Flow Diagram

```mermaid
flowchart LR
    A["Kaggle Insurance Dataset"] --> B["Data Cleaning"]
    B --> C["Exploratory Data Analysis"]
    C --> D["Feature Engineering"]
    D --> E["Model Training<br>(Linear, Ridge, Lasso, Random Forest)"]
    E --> F["Model Evaluation<br>(MAE, MSE, RMSE, R²)"]
    F --> G["Interpretation & Insights"]

    style A fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#000
    style B fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#000
    style C fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#000
    style D fill:#fde68a,stroke:#d97706,stroke-width:2px,color:#000
    style E fill:#fbcfe8,stroke:#be185d,stroke-width:2px,color:#000
    style F fill:#e9d5ff,stroke:#7e22ce,stroke-width:2px,color:#000
    style G fill:#fee2e2,stroke:#b91c1c,stroke-width:2px,color:#000