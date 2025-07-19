def engineer_features(df):
    df["Credit_Per_Month"] = df["Credit_Amount"] / df["Duration_Months"]
    df["Short_Term_Loan_Flag"] = df["Duration_Months"].apply(lambda x: 1 if x <= 6 else 0)
    df["No_Checking_Flag"] = df["Checking_Account_Status"].str.lower().apply(lambda x: 1 if "no checking" in x else 0)

    # Clean indicators
    if indicator_cols:
        df["Risk_Indicators_Sum"] = df[indicator_cols].sum(axis=1)
        df["Risk_Indicators_Sum"] = df[indicator_cols].sum(axis=1)

    # Savings category signal
    df["Low_Savings_Flag"] = (
        df["Savings_Account_Bonds"]
        .astype(str)
        .str.contains("<100|unknown", case=False)
        .astype(int)
    )
    return df