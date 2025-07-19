from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_catboost(X_train, y_train):
    model = CatBoostClassifier(verbose=0, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    # 🔁 Ensure columns are in same order as model expects
    if hasattr(model, "feature_names_"):
        X_test = X_test[model.feature_names_]
    elif hasattr(model, "feature_names_in_"):
        X_test = X_test[model.feature_names_in_]
    else:
        raise ValueError("Model does not expose feature names.")

    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    return {
        "AUC": roc_auc_score(y_test, probas),
        "Accuracy": accuracy_score(y_test, preds),
        "Report": classification_report(y_test, preds, output_dict=True)
    }