def patch_missing_inputs(df, required_cols, default_value="unknown"):
    for col in required_cols:
        if col not in df.columns:
            df[col] = default_value
    return df