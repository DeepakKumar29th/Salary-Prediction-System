# 💼 Salary Prediction System — Indian Job Market

A machine learning web application that predicts annual CTC for professionals in India based on their education, experience, job role, and location.

---

## 🌐 Live Demo

> https://salary-predict-system.streamlit.app/#salary-prediction-system

---

## 📸 Preview

> <img width="1183" height="797" alt="image" src="https://github.com/user-attachments/assets/2403c377-cf49-4df8-a863-bb654ec72c92" />

<img width="1071" height="861" alt="image" src="https://github.com/user-attachments/assets/4648d21c-db8a-4fa9-a822-443adb56f5de" />

---

## About the Project

One of the most common questions people have when entering the job market is — *"What salary should I expect?"*

This project tries to answer that. I built a salary prediction system trained on 11,000 employee records across 15 different industries in India. You fill in your degree, specialization, job role, company type, and city tier — and the model gives you an estimated annual salary based on those inputs.

The goal was to make it practical and realistic. So instead of just picking "Bachelor's" or "Master's", you can select your actual degree like *BCA with Data Science & AI specialization* or *MBA in Investment Banking* — because those details genuinely affect how much you earn.

---

## What It Can Do

- Predict annual salary in INR (Lakhs) based on your profile
- Supports 49 degree types across Bachelor's, Master's, and PhD
- Specialization matters — BCA General and BCA Data Science are treated differently
- Covers 15 job sectors and 120+ specific job roles
- City tier selection instead of individual city names (Tier 1 / 2 / 3)
- Shows a salary range and estimated monthly take-home

---

## 📊 Dataset

The dataset has 11,000 records and was built to reflect the 2026 Indian job market. It covers roles from Software Development, Data & AI, Finance & Banking, Healthcare, Consulting, and more. Salary ranges from around ₹3.9L to ₹98.9L with a median of ₹21.5L.

Features used for training:

| Feature | Type |
|---------|------|
| Education Degree | Categorical |
| Specialization | Categorical |
| Education Level | Categorical |
| Job Sector | Categorical |
| Job Role | Categorical |
| Company Size | Categorical |
| Work City Tier | Categorical |
| Years of Experience | Numerical |

---

## 🤖 Model

| Detail | Value |
|--------|-------|
| Algorithm | Gradient Boosting Regressor |
| Preprocessing | OneHotEncoding + StandardScaler |
| Train / Test Split | 80% / 20% |
| R² Score | ~0.94 |
| MAE | ~₹0.5 Lakhs |

A Scikit-learn Pipeline handles all preprocessing and prediction together, so the app loads fast and stays consistent.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas & NumPy | Data handling |
| Scikit-learn | ML model |
| Streamlit | Web app |
| OpenPyXL | Reading the dataset |
