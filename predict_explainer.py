import shap

def predict_with_explanation(model, X):
    explainer = shap.Explainer(model)
    shap_exp = explainer(X)  # SHAP Explanation object

    # For binary classification (multi-class SHAP format)
    if shap_exp.values.ndim == 3:
        explanation = shap.Explanation(
            values=shap_exp.values[0, 1],
            base_values=shap_exp.base_values[0, 1],
            data=shap_exp.data[0],
            feature_names=shap_exp.feature_names
        )
    else:
        explanation = shap_exp[0]  # For already single-row Explanation

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]

    # ✅ Return the correct variable name
    return pred, proba, explanation
