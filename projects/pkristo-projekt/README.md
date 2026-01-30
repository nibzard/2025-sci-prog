# Shipping Delay Prediction

---

## Introduction

In modern logistics and e-commerce, on-time delivery plays a critical role in customer satisfaction and cost efficiency.  
Delays in shipments can occur due to various operational and external factors such as mode of transport, product weight, discounts, and warehouse performance.

The goal of this project is to develop a **data-driven machine learning model** capable of **predicting whether a shipment will be delayed or delivered on time** based on its attributes.  
Such a model can help logistics companies optimize their processes and reduce overall delay rates.

---

## Hypothesis

Shipment delays can be accurately predicted using historical logistics data and a machine learning model trained on operational shipment features.

---

## Dataset

The dataset used in this project is the **Supply Chain Shipment Dataset** available on [Kaggle](https://www.kaggle.com/datasets/prachi13/customer-analytics).  
It contains more than 10,000 shipment records with both categorical and numerical attributes describing the shipping process.

Each record represents one shipment with the following attributes:

| Attribute | Description |
|------------|-------------|
| `Warehouse_block` | Warehouse from which the shipment was dispatched |
| `Mode_of_Shipment` | Delivery mode (Flight, Ship, Road) |
| `Customer_care_calls` | Number of customer care calls made for that shipment |
| `Customer_rating` | Rating provided by the customer (1–5) |
| `Cost_of_the_Product` | Product price in USD |
| `Prior_purchases` | Number of previous purchases by the customer |
| `Product_importance` | Product importance (low, medium, high) |
| `Gender` | Gender of the customer |
| `Discount_offered` | Discount applied to the product (%) |
| `Weight_in_gms` | Weight of the shipment (grams) |
| `Reached.on.Time_Y.N` | Target variable (1 = on time, 0 = delayed) |

---

## Methodology

1. **Data Preparation**
   - Load and inspect the dataset  
   - Handle missing values and remove irrelevant columns  
   - Encode categorical variables (`OneHotEncoder` / `LabelEncoder`)  
   - Create binary target variable `delay_flag`

2. **Exploratory Data Analysis (EDA)**
   - Analyze the distribution of delays by shipment mode and warehouse  
   - Visualize correlations between numerical variables (`Seaborn`, `Matplotlib`)  
   - Identify patterns and seasonal trends in shipment delays

3. **Feature Engineering**
   - Create additional features:
     - `is_heavy` – 1 if `Weight_in_gms` > average weight  
     - `discount_category` – low/medium/high discount level  
     - `shipment_value` – ratio of cost to weight

4. **Modeling**
   - Split data into training (80%) and testing (20%) sets  
   - Compare three models:
     - Logistic Regression  
     - Decision Tree Classifier  
     - Random Forest Classifier  

5. **Evaluation**
   - Evaluate models using:
     - Accuracy  
     - Precision  
     - Recall  
     - F1-score  
     - ROC-AUC  
   - Compare performance visually with bar charts and ROC curves  

6. **Interpretation**
   - Identify the most important features contributing to delay prediction  
   - Analyze which conditions (e.g., transport type, product weight, discount) most influence late deliveries  
   - Summarize findings and potential real-world applications

7. ## How to run
All results presented in this project can be reproduced by opening the notebook and executing all cells sequentially.


---

## Project Flow Diagram

```mermaid
flowchart LR
    A["Kaggle Shipment Dataset"] --> B["Data Cleaning"]
    B --> C["EDA"]
    C --> D["Feature Engineering"]
    D --> E["Model Training(Logistic Regression, Decision Tree, Random Forest)"]
    E --> F["Evaluation\Accuracy, Precision, Recall, F1, ROC-AUC)"]
    F --> G["Interpretation & Insights"]

    style A fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#000
    style B fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#000
    style C fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#000
    style D fill:#fde68a,stroke:#d97706,stroke-width:2px,color:#000
    style E fill:#fbcfe8,stroke:#be185d,stroke-width:2px,color:#000
    style F fill:#e9d5ff,stroke:#7e22ce,stroke-width:2px,color:#000
    style G fill:#fee2e2,stroke:#b91c1c,stroke-width:2px,color:#000
