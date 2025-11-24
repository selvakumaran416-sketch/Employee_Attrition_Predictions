import streamlit as st
import pandas as pd
import pickle
with open("C:/Users/Selva.M/Downloads/data_science/project_3/best_model.pkl","rb") as f:
    model = pickle.load(f)

with open("C:/Users/Selva.M/Downloads/data_science/project_3/preprocessing.pkl","rb") as f:
    preprocessing = pickle.load(f)

# Set the page config
st.set_page_config(page_title="EMPLOYEE ATTRITION DASHBOARD", layout="wide")

# Sidebar
with st.sidebar:
    st.title("EMPLOYEE PREDICTION")
    menu = st.radio("NAVIGATER", ["HOME", "EMPLOYEE ATTRITION ANALYSIS"])

# Home Page
if menu == "HOME":
    st.markdown("<h2 style='text-align: center;'>EMPLOYEE INSIGHTS DASHBOARD</h2>", unsafe_allow_html=True)
    st.info("VIEW HIGH RISK EMPLOYEES AND IMPORTANT KEY INSIGHTS TO REDUCE ATTRITION RATES")

    df = pd.read_csv("C:/Users/Selva.M/Downloads/data_science/project_3/cleaned_dataset.csv")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("HIGH RISKED EMPLOYEES")
        high_risk = df[df['Attrition'] > 0.8][['Age','TotalWorkingYears', 'Attrition']].head(10)
        st.dataframe(high_risk, use_container_width=True)

    with col2:
        st.markdown("HIGH JOB SATISFACTION EMPLOYEES")
        high_satisfaction = df[df['JobSatisfaction'] >= 4][['JobLevel', 'JobSatisfaction', 'Attrition']].head(10)
        st.dataframe(high_satisfaction, use_container_width=True)

    with col3:
        st.markdown("WORK LIFE BALANCED EMPLOYEES")
        life_balance = df[df['MonthlyIncome'] > 80][['JobRole', 'WorkLifeBalance', 'Attrition']].head(10)
        st.dataframe(life_balance, use_container_width=True)



# Prediction Page
elif menu == "EMPLOYEE ATTRITION ANALYSIS":
    st.markdown("<h1 style='text-align: center;'>ATTRITION PREDICTION</h1>", unsafe_allow_html=True)
    st.subheader("SELECT EMPLOYEE DETAILS TO PREDICT ATTRITION")

    with st.form("attrition_form"):
        BusinessTravel = st.selectbox("Business Travel", ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
        JobRole = st.selectbox("Job Role", ['Human Resources', 'Laboratory Technician', 'Healthcare Representative',
        'Sales Representative', 'Sales Executive', 'Manager', 'Manufacturing Director','Research Scientist', 'Research Director'])
        OverTime = st.selectbox("Over Time", ["Yes", "No"])
        Department = st.selectbox("Department", ["Research & Development", "Sales", "HR"])
        Gender = st.selectbox("Gender", ["Male", "Female"])
        MaritalStatus = st.selectbox("Marital Status", ["Married", "Divorced", "Single"])
        Age = st.number_input("Age", min_value=18, max_value=60, value=25)
        DistanceFromHome = st.number_input("Distance From Home (km)", min_value=1, max_value=30, value=5)
        EnvironmentSatisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
        JobInvolvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
        JobLevel = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        JobSatisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
        MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=10000)
        StockOptionLevel = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        TotalWorkingYears = st.number_input("Total Working Years", min_value=0, max_value=40, value=5)
        YearsAtCompany = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
        YearsInCurrentRole = st.number_input("Years in Current Role", min_value=0, max_value=40, value=5)
        YearsWithCurrManager = st.number_input("Years with Current Manager", min_value=0, max_value=40, value=5)
        WorkLifeBalance = st.selectbox("WorkLifeBalance", [1, 2, 3, 4])
        YearsSinceLastPromotion = st.selectbox("YearsSinceLastPromotion", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        Education = st.selectbox("Education", [1, 2, 3, 4])
        RelationshipSatisfaction = st.selectbox("RelationshipSatisfaction", [1, 2, 3, 4])
        NumCompaniesWorked = st.selectbox("NumCompaniesWorked", [1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
        PercentSalaryHike = st.selectbox("PercentSalaryHike", [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
        PerformanceRating = st.selectbox("PerformanceRating", [1, 2, 3, 4, 5])
        submitted = st.form_submit_button("Predict Attrition")

    if submitted:
        input_data = pd.DataFrame({
            "BusinessTravel": [BusinessTravel],
            "JobRole": [JobRole],
            "OverTime": [OverTime],
            "Department": [Department],
            "Gender": [Gender],
            "MaritalStatus": [MaritalStatus],
            "Age": [Age],
            "DistanceFromHome": [DistanceFromHome],
            "EnvironmentSatisfaction": [EnvironmentSatisfaction],
            "JobInvolvement": [JobInvolvement],
            "JobLevel": [JobLevel],
            "JobSatisfaction": [JobSatisfaction],
            "MonthlyIncome": [MonthlyIncome],
            "StockOptionLevel": [StockOptionLevel],
            "TotalWorkingYears": [TotalWorkingYears],
            "YearsAtCompany": [YearsAtCompany],
            "YearsInCurrentRole": [YearsInCurrentRole],
            "YearsWithCurrManager": [YearsWithCurrManager],
            'WorkLifeBalance':[WorkLifeBalance],
            'YearsSinceLastPromotion':[YearsSinceLastPromotion],
            'Education':[Education],
            'RelationshipSatisfaction':[RelationshipSatisfaction],
            'NumCompaniesWorked':[NumCompaniesWorked], 
            'PercentSalaryHike':[PercentSalaryHike], 
            'PerformanceRating':[PerformanceRating]
        })

        x_transformed = preprocessing.transform(input_data)
        prediction = model.predict(x_transformed)
        probability = model.predict_proba(x_transformed)[0][1]*100

        # Display results
        st.markdown("### PREDICTION RESULTS:")
        if prediction == 1:
            st.error(f"EMPLOYEE LIKES TO LEAVE THIS COMPANY (PROBABILITY: {probability:.2f})")
        else:
            st.success(f"EMPLOYEE LIKES TO STAY IN THIS COMPANY (PROBABILITY: {probability:.2f})")