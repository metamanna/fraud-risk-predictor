import pandas as pd
import os

import joblib
def load_background():
    return joblib.load("encoders/background.pkl")
def load_data():
    path = os.path.join("data", "german_credit_cleaned.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please ensure the file exists.")

    df = pd.read_csv(path)
    
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace(".", "", regex=False)

    # Convert categories to string
    cat_cols = [
        "Checking_Account_Status", "Credit_History", "Purpose", "Savings_Account_Bonds",
        "Employment_Since", "Personal_Status_Sex", "Other_Debtors", "Property",
        "Other_Installment_Plans", "Housing", "Job", "Telephone", "Foreign_Worker"
    ]
    df[cat_cols] = df[cat_cols].astype(str)

    return df