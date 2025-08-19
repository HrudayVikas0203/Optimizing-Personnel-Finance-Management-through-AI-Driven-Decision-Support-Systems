import streamlit as st
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
from utils.recommender import generate_advice

st.set_page_config(page_title="AI-Driven Personal Finance Advisor", layout="centered")

st.title("💰 AI-Driven Personal Finance Advisor")

# Load models
all_models = {f: joblib.load(f"models/{f}") for f in os.listdir("models") if f.endswith(".pkl") and f not in ["preprocessor.pkl"]}
preprocessor = joblib.load("models/preprocessor.pkl")

# User Inputs
st.subheader("Enter Your Financial Data")
income = st.number_input("Total Household Income", min_value=0)
food = st.number_input("Food Expenditure", min_value=0)
restaurants_hotels = st.number_input("Restaurants & Hotels", min_value=0)
special_occasions = st.number_input("Special Occasions", min_value=0)
alcohol = st.number_input("Alcoholic Beverages", min_value=0)
tobacco = st.number_input("Tobacco", min_value=0)
misc_goods = st.number_input("Miscellaneous Goods & Services", min_value=0)
clothing = st.number_input("Clothing & Footwear", min_value=0)
house_water = st.number_input("Housing & Water", min_value=0)
rental_value = st.number_input("Imputed House Rental Value", min_value=0)
medical = st.number_input("Medical Care", min_value=0)
transport = st.number_input("Transportation", min_value=0)
communication = st.number_input("Communication", min_value=0)
education = st.number_input("Education", min_value=0)
crop_farming = st.number_input("Crop Farming & Gardening", min_value=0)
entrepreneurial_income = st.number_input("Income from Entrepreneurial Activities", min_value=0)

marital_status = st.selectbox("Marital Status", ["Married", "Single"])
occupation = st.selectbox("Occupation", ["Engineer", "Teacher", "Doctor", "Business", "Lawyer"])

savings = st.number_input("Current Savings", min_value=0)

if st.button("Predict Savings & Get Advice"):
    # Build DataFrame with correct columns
    feature_df = pd.DataFrame([{
        "total_income": income,
        "food": food,
        "restaurants_hotels": restaurants_hotels,
        "special_occasions": special_occasions,
        "alcohol": alcohol,
        "tobacco": tobacco,
        "misc_goods": misc_goods,
        "clothing": clothing,
        "house_water": house_water,
        "rental_value": rental_value,
        "medical": medical,
        "transport": transport,
        "communication": communication,
        "education": education,
        "crop_farming": crop_farming,
        "entrepreneurial_income": entrepreneurial_income,
        "marital_status": marital_status,
        "occupation": occupation,
        "savings": savings,
        "savings_ratio": savings / (income + 1) if income > 0 else 0
    }])

    # Transform inputs
    features_processed = preprocessor.transform(feature_df)

    # ------------------ MODEL COMPARISON ------------------
    st.subheader("📊 Model Predictions Comparison")
    comparison_results = {}
    for model_name, model in all_models.items():
        comparison_results[model_name] = round(model.predict(features_processed)[0], 2)

    comp_df = pd.DataFrame(list(comparison_results.items()), columns=["Model", "Predicted Savings"])
    st.table(comp_df)

    # Highlight best prediction
    best_model = comp_df.loc[comp_df["Predicted Savings"].idxmax()]
    st.success(f"🏆 Best Predicted Savings: {best_model['Predicted Savings']} (by {best_model['Model']})")

    # ------------------ PERSONALIZED ADVICE ------------------
    expenses_dict = {
        "Food": food,
        "Restaurants & Hotels": restaurants_hotels,
        "Special Occasions": special_occasions,
        "Alcohol": alcohol,
        "Tobacco": tobacco,
        "Misc Goods & Services": misc_goods,
        "Clothing": clothing,
        "Housing & Water": house_water,
        "Rental Value": rental_value,
        "Medical": medical,
        "Transport": transport,
        "Communication": communication,
        "Education": education,
        "Crop Farming": crop_farming,
        "Entrepreneurial Income": entrepreneurial_income
    }

    advice = generate_advice(income, savings, expenses_dict)
    st.subheader("🤖 Personalized Financial Advice")
    st.write(advice)

    # ------------------ EXPENSE VISUALIZATION ------------------
    st.subheader("📊 Expense Breakdown")
    expenses_df = pd.DataFrame(list(expenses_dict.items()), columns=["Category", "Amount"])
    expenses_df = expenses_df[expenses_df["Amount"] > 0]

    if not expenses_df.empty:
        fig, ax = plt.subplots()
        ax.pie(expenses_df["Amount"], labels=expenses_df["Category"], autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)
    else:
        st.info("No expenses entered yet to show a breakdown.")
