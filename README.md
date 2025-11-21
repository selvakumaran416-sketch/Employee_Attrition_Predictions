🌟 Employee Attrition Analysis & Prediction

A Complete End-to-End Machine Learning & Streamlit Dashboard Project

📌 Overview

Employee attrition is a major challenge for organizations, directly impacting productivity, HR efforts, and replacement costs.
This project builds a machine learning–powered prediction system to identify employees at risk of leaving and provides insights via an interactive Streamlit Dashboard.

The solution includes:

A trained ML classification model

Data preprocessing pipeline

Interactive dashboard with analytics

Prediction page for real-time attrition probability

📑 Table of Contents

Project Highlights

Architecture Diagram

Tech Stack

Features

Installation

Project Structure

Model Details

How the Streamlit App Works

Future Enhancements

Author

🚀 Project Highlights
✔ Machine Learning Workflow

Data preprocessing

Exploratory Data Analysis (EDA)

Feature engineering

Model training & selection

Pickled ML model for deployment

✔ Streamlit Dashboard

Home Analytics Page

Prediction Page

✔ End-to-End Deployment Ready

Includes preprocessing.pkl and best_model.pkl

Cleaned dataset for dashboard analytics

🧱 Architecture Diagram
                     ┌────────────────────────┐
                     │     Raw Dataset.csv     │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │   Preprocessing & EDA   │
                     └────────────┬────────────┘
                                  │
                                  ▼
                ┌────────────────────────────────────┐
                │ Feature Engineering & ML Training  │
                └───────────────┬────────────────────┘
                                │
             ┌──────────────────┴──────────────────┐
             ▼                                     ▼
  ┌───────────────────────┐              ┌─────────────────────────┐
  │  preprocessing.pkl     │              │    best_model.pkl        │
  └───────────────────────┘              └─────────────────────────┘
             │                                     │
             └───────────────┬─────────────────────┘
                             ▼
                 ┌────────────────────────┐
                 │   Streamlit App (UI)   │
                 └────────────┬───────────┘
                              │
                              ▼
                ┌───────────────────────────────┐
                │ Prediction + Dashboard Output  │
                └───────────────────────────────┘

🛠 Tech Stack
Languages

Python 3.x

Libraries

Pandas

NumPy

Scikit-learn

Streamlit

Pickle

Tools

Jupyter Notebook

Streamlit

VS Code / PyCharm

⭐ Features
🏠 1. Dashboard Home Page

High-risk employees

High job satisfaction groups

Work-life balance insights

Clean tabular display

🧪 2. Attrition Prediction Page

Inputs 20+ employee features including:

Age

Department

Job Role

Overtime

Monthly Income

Work-Life Balance

Job Satisfaction

Years at Company

Promotion history

Output:

✔ "Likely to Leave" OR "Likely to Stay"

✔ Probability score

📊 3. ML Model

Random Forest / Logistic Regression

Encoded & scaled features

Feature selection applied

Saved via pickle

📁 4. Complete Codebase

Everything required to run and deploy the model.

⚙ Installation
1. Clone the repository
git clone https://github.com/your-username/employee-attrition.git
cd employee-attrition

2. Install dependencies
pip install -r requirements.txt

3. Run the Streamlit App
streamlit run Employee.py

📂 Project Structure
Employee-Attrition-Project/
│
├── Employee.py                 # Streamlit application
├── Employee_Attrition.ipynb    # EDA & model training notebook
├── best_model.pkl              # Trained Machine Learning model
├── preprocessing.pkl           # Preprocessing pipeline
├── cleaned_dataset.csv         # Final cleaned dataset
├── README.md                   # Documentation
└── requirements.txt            # Python dependencies

🤖 Model Details
Algorithms Tried

Logistic Regression

Decision Tree

Random Forest (final model)

Evaluation Metrics

Accuracy

Precision

Recall

F1-score

Confusion Matrix

ROC-AUC

Input Features

35+ employee attributes covering:

Demographics

Salary

Job role

Experience

Promotions

Satisfaction scores

Work-life balance

🖥 How the Streamlit App Works
Home Page

Displays:

High-risk employees

High job satisfaction groups

Work-life balance and income patterns

Prediction Page

User enters employee data

Data is transformed using preprocessing.pkl

Model predicts attrition using best_model.pkl

UI displays:

Prediction

Probability

🚀 Future Enhancements

🔹 Add SHAP feature importance
🔹 Add charts (pie chart, bar chart, heatmap)
🔹 Add employee filtering in dashboard
🔹 Add authentication to app
🔹 Deploy publicly using Render / Streamlit Cloud

👨‍💻 Author

SELVAKUMARAN M
Data Science & Analytics Enthusiast   
AA📜 LiceA
TAhis project is licensed under t  AAAAAAAAAAAAAAAAA
