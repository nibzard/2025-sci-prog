# Weather-Based Ski Resort Attendance Prediction

## Project Overview

This project analyzes how **snow conditions, calendar effects, and ski resort characteristics**
influence **ski resort attendance**.

Because real attendance data is not publicly available, attendance is modeled as a **proxy target**
(low / medium / high) derived from weather intensity, seasonality, and resort metadata.

The goal is to build a **reproducible machine learning pipeline** that predicts attendance levels
for ski resorts based on historical conditions.

---

## Research Questions

- How does snowfall intensity influence ski resort attendance?
- Are weekends and winter months associated with higher attendance?
- Which engineered features are most important for predicting attendance levels?
- How do linear models compare to tree-based models on this task?


---

## Data Sources

- **Snow data**: historical gridded snowfall data mapped to ski resort locations using spatial grid matching
- **Resort metadata**: location, altitude, number of slopes and lifts, country
- **Calendar features**: month, seasonality, weekend indicators

> ⚠️ Attendance is **synthetic (proxy)** and created for research and educational purposes only.

---

## Methodology

1. **Data collection & integration**

   - Mapping snowfall data to ski resorts
   - Aggregation at _resort × month_ level

2. **Exploratory Data Analysis (EDA)**

   - Proxy attendance distribution
   - Snow intensity and seasonality
   - Differences across countries and resorts

3. **Feature Engineering**

   - Seasonal indicators (winter / shoulder months)
   - Lagged snow features
   - Cyclical encoding of months (sin / cos)
   - Resort capacity and altitude features

4. **Modeling**

   Several classification models were trained and compared:

   - Baseline Logistic Regression: used as a simple and interpretable baseline model to establish a reference level of performance.

   - LinearSVC: a strong linear classifier well suited for high-dimensional, one-hot encoded feature spaces.

   - Random Forest Classifier: a non-linear ensemble model capable of capturing complex interactions between weather conditions, seasonality, and ski resort characteristics.

5. **Evaluation**
   - Train / test split
   - Accuracy and confusion matrices
   - Permutation feature importance

---

## Results

- The baseline Logistic Regression provides a simple reference point but is limited in capturing complex relationships.
- LinearSVC offers a strong and stable linear benchmark and performs well on high-dimensional feature representations.
- Random Forest captures non-linear interactions between features and achieves the best overall performance.
- Snow intensity, seasonality, and resort capacity features are among the most influential predictors.

---

## Limitations

- Attendance is a **proxy**, not real observed data.
- Snow is estimated from gridded datasets and mapped to resorts (possible spatial mismatch).
- Resort metadata may not reflect real-time operational conditions.

---

## Future Work

- Incorporate real attendance signals (ticket sales, holidays, Google Trends, social media).
- Extend analysis to multiple years and additional regions.
- Replace proxy target with real attendance data if available.

---

## Tools & Technologies

- **Python**
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

---

## Author

**Aneta Kalabric**
