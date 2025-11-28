# Customer Churn Insight Platform (IBM Telco)

A web application that provides customer churn analytics for telecom customers using the IBM Telco dataset. The application consists of a dashboard and an LLM chat interface that helps users identify key risk factors and explore insights interactively.

---

## 🎯 Objectives

- Analyze churn probability and churn risk over time using predictive and survival models.
- Identify influential features affecting customer retention.
- Provide explainable analytics through a visual dashboard.
- Enable natural language querying of churn insights using an LLM chat interface.

---

## 📊 Dataset

The [IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) is a dataset provided by IBM that contains information about the customers of a fictional telecom company.

The dataset contains customer demographics, service subscriptions, billing information, and churn labels.

---

## 🧰 Proposed Tech Stack

The tech stack is not yet finalized and is therefore subject to change.

| Component | Tool / Library |
|-----------|----------------|
| Frontend UI | React + Plotly |
| Backend API | FastAPI or Flask |
| LLM Integration | Google Gemini API |
| ML Models | scikit-learn |
| Survival Analysis | lifelines, scikit-survival |

---

## 📈 Planned Features

- 🔢 **Churn probability estimator**
- ⏱️ **Analysis of churn risk over time**
- 🔍 **Feature importance and customer risk factors**
- 🧾 **Segment analytics and comparison across customer types**
- 🤖 **LLM chat interface for question-based insights**  
  Example: *“Which contract type has the highest churn risk?”*