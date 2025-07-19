import sys, os
sys.path.append(os.path.dirname(__file__))

import streamlit as st
st.set_page_config(page_title="SHAP Explainability", layout="wide")
# Ensure session state key exists
if "pdf_file" not in st.session_state:
    st.session_state.pdf_file = None
st.markdown("""
<style>
    .reportview-container .main {
        background-color: #f9f9f9;
        padding: 20px;
    }
    .block-container {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import shap
import joblib
import random
import re
import streamlit.components.v1 as components

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

from modules.data_loader import load_data
from modules.model_trainer import train_random_forest
from modules.model_trainer import train_catboost  # already done for RF
from modules.predict_explainer import predict_with_explanation
from modules.input_utils import patch_missing_inputs
from modules.nlp_sentiment import get_sentiment_score
from modules.model_trainer import evaluate_model
from modules.transformer_sentiment import get_transformer_sentiment
from modules.bert_embedder import get_bert_embedding
from lime.lime_tabular import LimeTabularExplainer
from transformers import AutoTokenizer

# 🔠 Human-readable mappings from A11/A12/etc. to explanations
category_mappings = {
    "Checking_Account_Status": {
        "A11": "No checking account",
        "A12": "< 0 DM",
        "A13": "0 <= ... < 200 DM",
        "A14": ">= 200 DM or salary assignments"
    },
    "Savings_Account_Bonds": {
        "A61": "< 100 DM",
        "A62": "100 <= ... < 500 DM",
        "A63": "500 <= ... < 1000 DM",
        "A64": ">= 1000 DM",
        "A65": "Unknown / No savings"
    },
    "Purpose": {
        "A40": "Car (new)",
        "A41": "Car (used)",
        "A42": "Furniture/Equipment",
        "A43": "Radio/TV",
        "A44": "Domestic appliances",
        "A45": "Repairs",
        "A46": "Education",
        "A47": "Vacation",
        "A48": "Retraining",
        "A49": "Business",
        "A410": "Others"
    }
}

def get_model_metrics(df, model_features, text_only_features):
    from sklearn.model_selection import train_test_split

    # ➤ Prepare inputs
    X_full = df[model_features]
    y = df["Credit_Risk"]

    # ➤ Split full input
    X_train, X_val, y_train, y_val = train_test_split(X_full, y, stratify=y, test_size=0.2, random_state=42)
    X_val = X_val[model_features]  # For CatBoost compatibility

    # ➤ SMOTE resampling
    from imblearn.over_sampling import SMOTE
    X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_train, y_train)

    # ➤ Train models
    rf_model = train_random_forest(X_resampled, y_resampled)
    cb_model = train_catboost(X_resampled, y_resampled)

    # ➤ Evaluate
    rf_metrics = evaluate_model(rf_model, X_val, y_val)
    cb_metrics = evaluate_model(cb_model, X_val, y_val)

    # ➤ Justification-only model
    X_text = df[text_only_features]
    y_text = df["Credit_Risk"]
    X_train_txt, X_val_txt, y_train_txt, y_val_txt = train_test_split(X_text, y_text, stratify=y_text, test_size=0.2, random_state=42)
    text_model = train_random_forest(X_train_txt, y_train_txt)
    text_model_metrics = evaluate_model(text_model, X_val_txt, y_val_txt)
    auc_df = pd.DataFrame({
    "Model": ["Random Forest", "CatBoost"],
    "AUC Score": [rf_metrics["AUC"], cb_metrics["AUC"]]
    })

    return {
        "models": {
            "Random Forest": rf_model,
            "CatBoost": cb_model,
            "Text Only": text_model
        },
        "metrics": {
            "Random Forest": rf_metrics,
            "CatBoost": cb_metrics,
            "Text Only": text_model_metrics
        },
        "X_val": X_val,
        "y_val": y_val
    }


def explain_with_lime(model, instance, train_data, feature_names, class_names):
    explainer = LimeTabularExplainer(
        training_data=np.array(train_data),
        feature_names=feature_names,
        class_names=class_names,
        mode="classification"
    )
    exp = explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict_proba
    )
    return exp

# ✨ Cleaned encoding + feature creation
def engineer_features(df, is_train=True):
    from sklearn.preprocessing import LabelEncoder
    os.makedirs("encoders", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/predictions.csv"


    def _encode(col, fname):
        encoder_path = os.path.join("encoders", fname)
        if is_train:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            joblib.dump(le, encoder_path)
        else:
            le = joblib.load(encoder_path)
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else "Unknown")
            if "Unknown" not in le.classes_:
                le.classes_ = np.append(le.classes_, "Unknown")
            df[col] = le.transform(df[col])

    _encode("Checking_Account_Status", "le_checking.pkl")
    _encode("Savings_Account_Bonds", "le_savings.pkl")
    _encode("Purpose", "le_purpose.pkl")
    df["Credit_Per_Month"] = df["Credit_Amount"] / df["Duration_Months"]
    return df

# 🧾 Load + preview data
st.markdown("""
# 💳 Credit Risk Classifier Dashboard  
Empowering transparent credit decisions with explainable AI ⚖️🔍  
""")
st.markdown("""
> 📊 **Welcome to the Credit Risk Classifier Dashboard**

This app helps you:
- Predict creditworthiness of applicants 🧾
- Understand *why* a prediction is made (SHAP) 🔍
- Simulate sentiment impact on risk 🗣️
- View top features influencing past decisions 📊
""")
df = load_data()
# 🔁 Fix target encoding (only if needed)
# 📌 Define feature lists for hybrid input
numerical_features = ["Age", "Credit_Amount", "Duration_Months", "Credit_Per_Month"]
# text_features = [f"embed_{i}" for i in range(embed_dim)] + ["Sentiment_Score"]
# model_features = numerical_features + text_features
if df["Credit_Risk"].min() == 1:
    df["Credit_Risk"] = df["Credit_Risk"].map({1: 0, 2: 1})  # 0: Good, 1: Bad

st.header("📂 Dataset Preview")
st.dataframe(df.head())
# Placeholder in case batch prediction section runs before model setup
selected_model_name = "Random Forest"  # or any default fallback
# 📤 Allow Batch Prediction via CSV Upload
uploaded_file = st.sidebar.file_uploader("📤 Upload CSV for Batch Predictions", type="csv")
confidence_threshold = st.sidebar.slider(
            "Confidence Threshold", 0.5, 0.95, 0.75, 0.01,
            help="Set threshold for high confidence label"
        )

if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(batch_df.head())

    # TODO: Apply same preprocessing
    st.info("⚠️ Batch prediction not yet implemented. Apply same preprocessing pipeline here.")
    try:
        # Step 1: Feature engineering
        batch_df = engineer_features(batch_df, is_train=False)

        # Step 2: Generate sentiment scores
        batch_df["Sentiment_Score"] = batch_df["Justification"].apply(cached_sentiment)

        # Step 3: Generate BERT embeddings
        embed_vectors = batch_df["Justification"].apply(cached_embedding)
        embed_df = pd.DataFrame(embed_vectors.tolist(), columns=[f"embed_{i}" for i in range(embed_vectors[0].shape[0])])

        # Step 4: Combine
        batch_df = pd.concat([batch_df.reset_index(drop=True), embed_df], axis=1)
        batch_df["Credit_Per_Month"] = batch_df["Credit_Amount"] / batch_df["Duration_Months"]

        # Step 5: Patch missing
        batch_df = patch_missing_inputs(batch_df, model_features, default_value=0)
        batch_df = batch_df[model_features]

        # Step 6: Predict
        selected_batch_model = get_trained_models()[selected_model_name]
        batch_preds = selected_batch_model.predict(batch_df)
        batch_probs = selected_batch_model.predict_proba(batch_df)[:, 1]

        # Step 7: Display
        result_df = pd.DataFrame({
            "Prediction": ["Bad" if p == 1 else "Good" for p in batch_preds],
            "Probability": batch_probs
        })
        final_output = pd.concat([batch_df.reset_index(drop=True), result_df], axis=1)
        st.subheader("📊 Batch Prediction Results")
        st.dataframe(final_output)

        # Optional: Download results
        st.download_button(
            label="📥 Download Results CSV",
            data=final_output.to_csv(index=False).encode(),
            file_name="batch_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Failed to process batch predictions: {e}")


with st.expander("📊 Overview Summary"):
    if os.path.exists("logs/predictions.csv"):
        df_logs = pd.read_csv("logs/predictions.csv", on_bad_lines='skip')
        st.metric("📈 Total Predictions", len(df_logs))
        st.metric("🔎 Avg Confidence", f"{df_logs['Probability'].mean():.2%}")
        
        fig, ax = plt.subplots()
        sns.countplot(x="Prediction", data=df_logs, palette="coolwarm", ax=ax)
        ax.set_xticklabels(["Good", "Bad"])
        ax.set_title("Predicted Risk Distribution")
        st.pyplot(fig)

col1, col2 = st.columns(2)
with col1:
    st.metric("Total Records", len(df))
    st.metric("Missing Values", df.isnull().sum().sum())
    st.metric("Duplicates", df.duplicated().sum())

with col2:
    st.write("🧾 Available columns:", df.columns.tolist())
    st.write("### Credit Risk Distribution")
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(x="Credit_Risk", hue="Credit_Risk", data=df, palette="Set2", ax=ax, legend=False)
    ax.set_title("Target Class Balance")
    st.pyplot(fig)

df = engineer_features(df, is_train=True)

# 📊 EDA
st.write("### 💸 Credit Amount vs Credit Risk")
fig1, ax1 = plt.subplots()
sns.boxplot(x="Credit_Risk", y="Credit_Amount", data=df, palette="Set2", ax=ax1)
st.pyplot(fig1)

st.write("### 📈 Engineered Feature: Credit_Per_Month")
fig2, ax2 = plt.subplots()
sns.histplot(df["Credit_Per_Month"], kde=True, bins=30, ax=ax2, color="skyblue")
st.pyplot(fig2)

st.write("### 🧊 Correlation Matrix")
fig3, ax3 = plt.subplots(figsize=(10,6))
num_cols = df.select_dtypes(include="number").columns
sns.heatmap(df[num_cols].corr(), annot=True, cmap="YlGnBu", fmt=".2f", ax=ax3)
st.pyplot(fig3)

# 🔮 Train model
st.title("🔍 Credit Risk Explanation")
# df["Justification"] = df["Credit_Risk"].apply(
#     lambda x: "I need it for home renovation" if x == 0 else "I lost my job and need urgent help"
# )
import random

justification_map = {
    0: [
        "I need it for home renovation",
        "To buy a new car",
        "For education expenses",
        "Planning a vacation",
        "Need funds for business expansion",
        "To pay medical bills",
        "Renovating my home office"
    ],
    1: [
        "I lost my job and need urgent help",
        "Medical emergency at home",
        "Loan needed due to critical illness",
        "Family emergency, require urgent funds",
        "Business suffered huge losses",
        "Struggling with unemployment"
    ]
}

df["Justification"] = df["Credit_Risk"].apply(lambda x: random.choice(justification_map[x]))
df["Sentiment_Score"] = df["Justification"].apply(get_transformer_sentiment)

# 🔍 JUSTIFICATION-ONLY MODEL (Phase 4 - Hybrid Evaluation)
st.subheader("📊 Justification-Only Model Evaluation (Experimental)")

expected_embed_cols = [f"embed_{i}" for i in range(768)]
available_embed_cols = [col for col in expected_embed_cols if col in df.columns]

if "Sentiment_Score" in df.columns:
    available_embed_cols.append("Sentiment_Score")

X_text_only = df[available_embed_cols]
y_text_only = df["Credit_Risk"]

X_train_txt, X_val_txt, y_train_txt, y_val_txt = train_test_split(
    X_text_only, y_text_only, stratify=y_text_only, test_size=0.2, random_state=42
)

# Step 2: Generate embeddings from justification
embeddings = df["Justification"].apply(get_bert_embedding)

# Step 3: Convert to columns
embed_dim = embeddings.iloc[0].shape[0] if hasattr(embeddings.iloc[0], 'shape') else len(embeddings.iloc[0])
embed_df = pd.DataFrame(embeddings.tolist(), columns=[f"embed_{i}" for i in range(embed_dim)])
text_features = [f"embed_{i}" for i in range(embed_dim)] + ["Sentiment_Score"]
model_features = numerical_features + text_features


# Step 4: Merge with original df
df = pd.concat([df.reset_index(drop=True), embed_df], axis=1)
# sentiment = max(min(sentiment, 1), -1)  # clip between -1 and 1
# X = df[[f"embed_{i}" for i in range(768)] + ["Sentiment_Score"]]  # if using sentiment too
X = df[model_features]
y = df["Credit_Risk"]
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_val = X_val[model_features]  # ✅ Ensure correct order for CatBoost
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_train, y_train)

st.cache_resource.clear()

# ✅ Cached transformer-based sentiment function
@st.cache_data
def cached_sentiment(text):
    return get_transformer_sentiment(text)

@st.cache_resource
def get_trained_models():
    rf_model = train_random_forest(X_resampled[model_features], y_resampled)
    cb_model = train_catboost(X_resampled[model_features], y_resampled)
    if "model_features" not in globals():
        st.warning("⚠️ Please train the model first by running a prediction below.")
        st.stop()
    return {"Random Forest": rf_model, "CatBoost": cb_model}

# Load BERT tokenizer once
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def is_gibberish(text):
    if not text or len(text.strip()) < 3:
        return True

    tokens = tokenizer.tokenize(text)
    
    # ✅ Accept if at least one token is meaningful
    if all(t == '[UNK]' for t in tokens):
        return True
    
    # ✅ Allow short but valid phrases like "education loan"
    if len(tokens) < 2:
        words = text.lower().split()
        valid_keywords = ["loan", "education", "medical", "home", "repair", "business", "vehicle", "travel", "study", "marriage", "bills"]
        if any(word in valid_keywords for word in words):
            return False
        return True

    return False

st.write("🧮 Resampled Class Distribution (Train):", pd.Series(y_resampled).value_counts().to_dict())

BACKGROUND = shap.sample(X_resampled[model_features], 100)
joblib.dump(BACKGROUND, "encoders/background.pkl")
st.write("🔍 Background shape:", BACKGROUND.shape)

# Define features
text_only_features = [f"embed_{i}" for i in range(embed_dim)] + ["Sentiment_Score"]

results = get_model_metrics(df, model_features, text_only_features)

rf_metrics = results["metrics"]["Random Forest"]
cb_metrics = results["metrics"]["CatBoost"]
text_model_metrics = results["metrics"]["Text Only"]

X_val = results["X_val"]
y_val = results["y_val"]

# ✅ Create auc_df BEFORE plotting
auc_df = pd.DataFrame({
    "Model": ["Random Forest", "CatBoost", "Text Only"],
    "AUC Score": [rf_metrics["AUC"], cb_metrics["AUC"], text_model_metrics["AUC"]]
})

# ✅ Plot AUC Comparison
fig_auc, ax_auc = plt.subplots()
sns.barplot(data=auc_df, x="Model", y="AUC Score", palette="coolwarm")
ax_auc.set_title("AUC Comparison: Full vs Justification-Only Model")
plt.xticks(rotation=0)
st.pyplot(fig_auc)

# # 📊 Show AUC compariso

from sklearn.metrics import classification_report
# ✅ Extract models from results
rf_model = results["models"]["Random Forest"]
cb_model = results["models"]["CatBoost"]
text_model = results["models"]["Text Only"]

# ✅ Make model_dict available for selection
model_dict = {
    "Random Forest": rf_model,
    "CatBoost": cb_model,
    "Text Only": text_model
}

st.subheader("🧾 Classification Report (Validation Set)")

st.markdown("**Random Forest**")
st.text(classification_report(y_val, rf_model.predict(X_val)))

st.markdown("**CatBoost**")
st.text(classification_report(y_val, cb_model.predict(X_val)))

st.markdown("## 📊 Project Summary")
col1, col2, col3 = st.columns(3)

col1.metric("📦 Dataset Size", f"{len(df)} rows")
col2.metric("⚖️ Class Balance", f"{df['Credit_Risk'].value_counts().to_dict()}")
col3.metric("🧠 Models Available", ", ".join(model_dict.keys()))


# ✅ Model Selection UI
st.subheader("⚙️ Choose Model")

st.subheader("📊 Model AUC Comparison")
with st.expander("📈 Compare ROC Curves"):
    fig, ax = plt.subplots()
    from sklearn.metrics import RocCurveDisplay
    RocCurveDisplay.from_estimator(rf_model, X_val, y_val, ax=ax, name="Random Forest")
    RocCurveDisplay.from_estimator(cb_model, X_val, y_val, ax=ax, name="CatBoost")
    ax.set_title("ROC Curve Comparison")
    st.pyplot(fig)

st.dataframe(auc_df, use_container_width=True)

feature_columns = X.columns.tolist()
df = pd.concat([df.reset_index(drop=True), embed_df], axis=1)


# 📝 User Input
st.subheader("📋 Applicant Details")
age = st.slider("Age", 18, 75, 30, help="Applicant's age in years")
amount = st.number_input("Credit Amount", min_value=0, value=1200, help="Loan amount you are requesting")
if amount > 5000 and age < 21:
    st.warning("⚠️ High loan amount requested by a very young applicant.")
if amount > 10000:
    st.error("❌ Loan amount exceeds system limit. Please lower it.")
duration = st.slider("Duration (months)", 6, 60, 24, help="Time over which the loan will be repaid")
purpose = st.selectbox("Purpose", df["Purpose"].unique(), help="The reason for applying for the credit")

user_dict = {
    "Age": age,
    "Credit_Amount": amount,
    "Duration_Months": duration,
    "Purpose": purpose,
    "Checking_Account_Status": "no_checking",
    "Savings_Account_Bonds": "no_savings"
}

input_df = pd.DataFrame([user_dict])
required_cols = X.columns.tolist()  # X = the training data with all feature names
# input_df = patch_missing_inputs(input_df, required_cols, default_value=0)
input_df = patch_missing_inputs(input_df, model_features, default_value=0)
input_df = input_df[model_features]

st.write("📊 Input Data")
st.dataframe(input_df)

st.subheader("⚙️ Choose Model")
selected_model_name = st.selectbox("🤖 Select Model", list(model_dict.keys()))
selected_model = model_dict[selected_model_name]
st.header("🗣️ Applicant Justification")

user_text = st.text_area(
    "Why do you need this credit?",
    placeholder="e.g. I need it for medical bills...",
    help="This helps analyze your intent using sentiment analysis"
)
@st.cache_data
def cached_embedding(text):
    return get_bert_embedding(text)

if st.button("Analyze Justification"):
    embedding_vector = cached_embedding(user_text)
    sentiment = cached_sentiment(user_text)

if not user_text or len(user_text.strip()) < 5:
    st.warning("⚠️ Please enter a minimum 4 word meaningful justification to get an accurate sentiment score.")
    st.stop()

if is_gibberish(user_text):
    try:
        from langdetect import detect
        if detect(user_text) != 'en':
            st.warning("⚠️ Please write your justification in English for accurate analysis.")
    except:
        pass  # Optional fallback if langdetect not installed
    st.error("❌ Your input appears to be invalid or nonsensical. Please provide a meaningful sentence.")
    st.stop()

def adjust_sentiment_score(text: str, original_score: float) -> float:
    if is_gibberish(text):
        return -1.0  # Force negative score for nonsensical inputs

    text_lower = text.lower()
    word_count = len(text_lower.split())

    neutral_keywords = ["loan", "requirement", "purchase", "renovation", "education", "investment", "business", "travel"]
    negative_keywords = ["urgent", "emergency", "unemployed", "hospital", "lost job", "critical", "medical"]

    if word_count <= 4 and not any(neg in text_lower for neg in negative_keywords):
        return 0.0  # Force neutral for plain short messages

    return original_score

@st.cache_data
def cached_sentiment(text):
    return get_transformer_sentiment(text)

@st.cache_data
def cached_bert_embedding(text):
    return get_bert_embedding(text)

@st.cache_data
def cached_embedding(text):
    return get_bert_embedding(text)

# Step 1: Gibberish check (early stop or manual override)
if is_gibberish(user_text):
    sentiment = -1.0  # Force very negative
else:
    # Step 2: Get transformer sentiment
    raw_sentiment = cached_sentiment(user_text)
    sentiment = adjust_sentiment_score(user_text, raw_sentiment)


# Step 3: Clip to range
sentiment = max(min(sentiment, 1), -1)

# Step 4: Get sentiment label
if sentiment > 0.1:
    sentiment_label = "Positive"
elif sentiment < -0.1:
    sentiment_label = "Negative"
else:
    sentiment_label = "Neutral"

def build_input_vector(user_dict, user_text):
    input_df = pd.DataFrame([user_dict])
    embed_vector = cached_embedding(user_text)
    for i, val in enumerate(embed_vector):
        input_df[f"embed_{i}"] = val
    input_df["Sentiment_Score"] = cached_sentiment(user_text)
    input_df = patch_missing_inputs(input_df, required_cols, default_value=0)
    return input_df

# st.markdown(f"🗣️ **Sentiment Label:** `{sentiment_label}`")

if sentiment_label == "Positive":
    st.success("🙂 Your justification has a positive tone.")
elif sentiment_label == "Negative":
    st.error("⚠️ Your justification has a negative tone and may increase risk.")
else:
    st.info("ℹ️ Your justification appears neutral.")


# Now it is safe to use user_text
sentiment = max(min(sentiment, 1), -1)  # safely clip between -1 and 1
input_df["Sentiment_Score"] = sentiment
raw_user_inputs = input_df.copy()  # ✅ Save original inputs for What-If section
input_df = input_df[model_features]  # ✅ includes Age, Credit_Amount etc. in right order
st.write("📊 Sentiment Score:", sentiment)
sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
st.markdown(f"🗣️ **Sentiment Label:** `{sentiment_label}`")

X = df.drop("Credit_Risk", axis=1)
y = df["Credit_Risk"]

from fpdf import FPDF
from io import BytesIO

def generate_pdf_report(input_df, prediction, probability, sentiment_score, shap_values, feature_names, user_justification):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Credit Risk Prediction Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=10)
    for col in input_df.columns:
        val = input_df[col].values[0]
        readable_reason = explanation_dict.get(col, col)
        # pdf.cell(200, 8, txt=f"{readable_reason}: {val}", ln=True)
    
    # 🌐 Apply readable mapping if available
    if col in category_mappings:
        val_map = category_mappings[col]
        val = next((v for k, v in val_map.items() if input_df[col].values[0] == k or input_df[col].values[0] == v), val)

    pdf.cell(200, 8, txt=f"{col}: {val}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(200, 8, txt=f"Prediction: {'Bad Risk' if prediction == 1 else 'Good Risk'}", ln=True)
    pdf.cell(200, 8, txt=f"Probability: {probability:.2%}", ln=True)
    pdf.cell(200, 8, txt=f"Sentiment Score: {sentiment_score}", ln=True)
    pdf.multi_cell(0, 8, txt=f"Justification: {user_justification}", align='L')

    pdf.ln(5)
    pdf.cell(200, 8, txt="Top Feature Contributions (SHAP):", ln=True)
    # 🔢 Extract top contributing features before logging
    sorted_idx = np.argsort(np.abs(shap_values))[::-1]
    top_n = min(3, len(shap_values))
    top_idx = sorted_idx[:top_n]


    pdf_output = pdf.output(dest='S').encode('latin1')
    return BytesIO(pdf_output)

if st.button("Explain Prediction"):
    input_df = input_df[model_features]  # ✅ Important
    pred, proba, shap_vals = predict_with_explanation(selected_model, input_df)
    # 🟢 Add confidence color band
    if proba >= confidence_threshold:
        conf_label = "🟢 High Confidence"
    elif proba >= 0.55:
        conf_label = "🟠 Medium Confidence"
    else:
        conf_label = "🔴 Low Confidence"

    features = shap_vals.feature_names
    if "Sentiment_Score" in features:
        sent_idx = features.index("Sentiment_Score")
        shap_values_flat = np.array(shap_vals.values).flatten()  # explanation for single input row
    if sent_idx < len(shap_values_flat):
        if abs(shap_values_flat[sent_idx]) > 0.1:
            st.info("🧠 Sentiment had a major influence on the model’s decision.")
    else:
        st.warning("⚠️ SHAP values not aligned with features — skipping sentiment influence check.")


    st.markdown(f"### {conf_label} (`{proba:.2%}`)")
    if proba >= 0.75:
        bar_color = "#28a745"  # Green
    elif proba >= 0.55:
        bar_color = "#fd7e14"  # Orange
    else:
        bar_color = "#dc3545"  # Red

    # Render a horizontal bar using HTML
    components.html(f"""
    <div style='width: 100%; background-color: #e0e0e0; border-radius: 8px; margin-top: 10px;'>
    <div style='width: {proba*100:.1f}%; background-color: {bar_color}; padding: 8px 0; border-radius: 8px; text-align: center; color: white; font-weight: bold;'>
        Confidence: {proba*100:.1f}%
    </div>
    </div>
    """, height=50)

    shap_values = shap_vals.values.flatten()
    features = shap_vals.feature_names

    # Show neutral sentiment warning
    if sentiment == 0:
        st.warning("⚠️ Sentiment was neutral. Prediction might not reflect true justification.")
        # st.stop()


    # Extract and log top SHAP features
    sorted_idx = np.argsort(np.abs(shap_values))[::-1]
    top_n = min(3, len(shap_values))
    top_idx = sorted_idx[:top_n]

    log_row = input_df.copy()
    log_row["Prediction"] = pred
    log_row["Probability"] = proba
    log_row["Top_3_Features"] = ", ".join([features[i] for i in top_idx])
    log_row["Justification"] = user_text.replace(",", ";") if user_text else "Not provided"

    log_path = "logs/predictions.csv"
    if not os.path.exists(log_path):
        log_row.to_csv(log_path, index=False)
        st.write("🔄 Log Written (with headers):")
        st.dataframe(log_row)
    else:
        log_row.to_csv(log_path, mode="a", header=False, index=False)
        st.write("🔄 Log Appended (no header):")
        st.dataframe(log_row)

    if pred == 1:
        st.error(f"🔴 **Bad Risk** ({proba:.1%})")
    else:
        st.success(f"🟢 **Good Risk** ({proba:.1%})")
    risk_emoji = "🟢" if pred == 0 else "🔴"
    risk_label = "Bad Risk" if pred == 1 else "Good Risk"
    st.markdown(f"### {risk_emoji} **Prediction:** `{risk_label}` ({proba:.1%})")


    # 👤 Plain Explanation
    st.markdown("### 🧠 Explanation in Simple Terms")
    key_reason = features[top_idx[0]]
    direction = "increased" if shap_values[top_idx[0]] > 0 else "decreased"
    risk_type = "Bad" if pred == 1 else "Good"

    # Add this FIRST (before calling .get())
    explanation_dict = {
        "Sentiment_Score": "Tone of justification message",
        "Credit_Amount": "Requested loan amount",
        "Duration_Months": "Repayment period in months",
        # You can add more like:
        "Credit_Per_Month": "Loan burden per month",
        "Age": "Applicant's age"
    }

    readable_reason = explanation_dict.get(key_reason, key_reason)
    st.info(
        f"The model predicted **{risk_type} risk** mainly because `{key_reason}` **{direction} your risk** score."
    )
    explanation_dict = {
    "Sentiment_Score": "Tone of justification message",
    "Credit_Amount": "Requested loan amount",
    "Duration_Months": "Repayment period in months",
    # Add others if needed
    }
    readable_reason = explanation_dict.get(key_reason, key_reason)
    st.markdown(f"🧠 This means: **{readable_reason}** influenced the decision.")


    st.info("🔑 Key Factors Influencing Prediction:")
    for idx in top_idx:
        st.markdown(f"- **{features[idx]}**: {shap_values[idx]:+.2f}")
    def generate_reason_summary(top_features, shap_values):
        summary = []
        for i in top_features:
            val = shap_values[i]
            trend = "increased" if val > 0 else "decreased"
            readable = explanation_dict.get(features[i], features[i])
            summary.append(f"{readable} {trend} the risk")
        return "; ".join(summary)

    st.markdown("### 🧾 Model Summary")
    st.success(generate_reason_summary(top_idx, shap_values))


    # Debug
    with st.expander("🔍 SHAP Debug Info"):
        st.write("Type:", type(shap_vals))
        st.write("Shape of values:", shap_vals.values.shape)   # Should be (n_features,)
        st.write("Base value:", shap_vals.base_values)
        st.write("Feature names:", shap_vals.feature_names)

    # ✅ SHAP Waterfall
    st.subheader("📉 SHAP Waterfall")
    shap.plots.waterfall(shap_vals, show=False)
    st.pyplot(plt.gcf())

    # ✅ SHAP Bar (custom, already working for you)
    st.subheader("📊 SHAP Bar Chart")
    with st.expander("🟡 LIME Explanation (Local)"):
        try:
            lime_exp = explain_with_lime(
                model=selected_model,
                instance=input_df.iloc[0].values,
                train_data=X_train,
                feature_names=X_train.columns.tolist(),
                class_names=["Good", "Bad"]
            )

            lime_fig = lime_exp.as_pyplot_figure()
            st.pyplot(lime_fig)
        except Exception as e:
            st.warning(f"LIME could not explain this instance: {e}")

    with st.expander("📉 SHAP Decision Plot"):
        try:
            from shap import Explanation

            feature_names = input_df.columns.tolist()
            shap_fixed = Explanation(
                values=shap_vals.values,
                base_values=shap_vals.base_values,
                data=input_df.values,
                feature_names=feature_names
            )

            shap.plots.decision(shap_fixed, show=False)
            st.pyplot(plt.gcf())

        except Exception as e:
            st.warning(f"Couldn't render SHAP decision plot: {e}")


    shap_values = shap_vals.values
    features = shap_vals.feature_names
    sorted_idx = np.argsort(np.abs(shap_values))[::-1]
    top_n = min(6, len(shap_values))
    top_idx = sorted_idx[:top_n]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['green' if val >= 0 else 'red' for val in shap_values[top_idx]]
    ax.barh(range(top_n), shap_values[top_idx], color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([features[i] for i in top_idx])
    ax.set_xlabel("SHAP Value")
    ax.set_title("Top Feature Contributions")
    for i in top_idx:
        feat = features[i]
        val = shap_values[i]
        reason = explanation_dict.get(feat, feat)
        # pdf.cell(200, 8, txt=f"{reason}: {val:+.2f}", ln=True)
    ax.invert_yaxis()

    for i, val in enumerate(shap_values[top_idx]):
            ax.text(val, i, f"{val:+.2f}", va='center',
                    ha='left' if val >= 0 else 'right', color='black')

    st.pyplot(fig)
    pdf_file = generate_pdf_report(input_df, pred, proba, sentiment, shap_values, features, user_text)
    # ✅ This goes at the end of the "Explain Prediction" block
    if pdf_file:
        st.markdown("### 📥 Download Your Prediction Report")
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_file,
            file_name="credit_risk_report.pdf",
            mime="application/pdf",
            key="download_main_report"  # Unique key to avoid widget duplication
        )

    st.session_state.pdf_file = pdf_file
    # ⬇️ Add this right after the confidence section (outside Explain Prediction block)
if "pdf_file" in st.session_state and st.session_state.pdf_file is not None:
    st.markdown("### 📄 Download Your Prediction Report")
    st.download_button(
    label="📄 Download PDF Report",
    data=st.session_state.pdf_file,
    file_name="credit_risk_report.pdf",
    mime="application/pdf",
    key="session_pdf_download"
    )


with st.expander("🧹 Admin Tools (One-Time Cleanup)", expanded=False):
    if st.button("Delete Old predictions.csv Log File"):
        if os.path.exists("logs/predictions.csv"):
            os.remove("logs/predictions.csv")
            st.success("🧹 Old log file deleted successfully.")
        else:
            st.info("ℹ️ Log file does not exist.")
log_path = "logs/predictions.csv"
if os.path.exists(log_path):
    df_logs = pd.read_csv(log_path, on_bad_lines='skip')  # For pandas >= 1.3

    st.dataframe(df_logs.tail(10), use_container_width=True)
    if "Probability" in df_logs.columns:
        st.subheader("📈 Prediction Confidence Over Time")
        st.sidebar.markdown("## 📊 Trends Summary")
        fig, ax = plt.subplots()
        df_logs["Probability"].tail(20).plot(kind="line", marker="o", ax=ax)
        ax.set_title("Recent Prediction Probabilities")
        ax.set_ylabel("Probability of Predicted Class")
        st.pyplot(fig)

        if st.sidebar.button("🔁 Refresh Models"):
            get_trained_models.clear()
            st.success("Models reloaded.")

        # ✅ Show in main area (optional visual preview)
        st.markdown("### 📄 Report Ready in Sidebar")

        # ✅ Always show the sidebar block
        with st.sidebar:
            st.markdown("## 📥 Download Report")
            if "pdf_file" in st.session_state and st.session_state.pdf_file is not None:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=st.session_state.pdf_file,
                    file_name="credit_risk_report.pdf",
                    mime="application/pdf",
                    key="download_sidebar_report"  # ✅ Unique key

                )
            else:
                st.info("ℹ️ Run a prediction to generate and download the PDF report.")



    if "Top_3_Features" in df_logs.columns:
        st.write("✅ Top Features from Recent Predictions:")
        st.dataframe(df_logs[["Top_3_Features", "Justification"]].tail(5))

    st.subheader("📊 Prediction Class Balance")
    fig, ax = plt.subplots()
    sns.countplot(x="Prediction", data=df_logs, palette="Set2")
    ax.set_xticklabels(["Good", "Bad"])
    ax.set_title("Model Predictions Over Time")
    st.pyplot(fig)

else:
    st.info("ℹ️ No predictions logged yet.")

with st.expander("🎛️ Sentiment Impact Test"):
    st.caption("ℹ️ This simulates how your justification's emotional tone (negative to positive) affects the risk prediction.")
    test_sent = st.slider("Simulate Sentiment", -1.0, 1.0, 0.0, step=0.1)
    # input_df["Sentiment_Score"] = test_sent
    pred, proba, _ = predict_with_explanation(selected_model, input_df)  # ✅ Fix
    risk_label = "Bad Risk" if pred == 1 else "Good Risk"
    st.markdown(f"🔁 **Simulated Sentiment:** `{test_sent:+.2f}`<br>📊 **Predicted:** `{risk_label}` ({proba:.2%})", unsafe_allow_html=True)

    st.write("📌 Final Input Columns:", input_df.columns.tolist())
    st.write("📌 Final Input Values:", input_df.iloc[0].to_dict())

with st.expander("🧪 What-If Scenario Playground"):
    st.markdown("Try adjusting key inputs to see how the prediction changes.")

    # Start from original inputs
    whatif_df = raw_user_inputs.copy()

    # 🔧 Modifiable Inputs
    st.markdown("#### 🔧 Adjust Inputs")
    whatif_df["Age"] = st.slider("🧑 Age", 18, 75, int(whatif_df["Age"].values[0]), help="Applicant's age", key="whatif_age")
    whatif_df["Credit_Amount"] = st.slider("💰 Credit Amount", 0, 10000, int(whatif_df["Credit_Amount"].values[0]), help="Requested loan amount", key="whatif_amount")
    whatif_df["Duration_Months"] = st.slider("📆 Duration", 6, 60, int(whatif_df["Duration_Months"].values[0]), key="whatif_duration")
    whatif_df["Sentiment_Score"] = st.slider("🗣️ Sentiment Score", -1.0, 1.0, float(whatif_df["Sentiment_Score"].values[0]), step=0.1, key="whatif_sentiment")

    # Derived feature
    whatif_df["Credit_Per_Month"] = whatif_df["Credit_Amount"] / whatif_df["Duration_Months"]

    # 🔁 Embed justification again
    embed_vector = get_bert_embedding(user_text)
    for i, val in enumerate(embed_vector):
        whatif_df[f"embed_{i}"] = val

    # ✅ Keep only model input columns
    whatif_df = patch_missing_inputs(whatif_df, model_features, default_value=0)
    whatif_df = whatif_df[model_features]

    # 🎯 Original
    st.markdown("#### 🎯 Original Prediction")
    orig_pred, orig_proba, orig_shap = predict_with_explanation(selected_model, whatif_df)
    orig_label = "Bad Risk" if orig_pred == 1 else "Good Risk"
    st.write(f"🔎 **Original Prediction:** `{orig_label}` ({orig_proba:.2%})")

    # 🔄 Updated Prediction (same as original here, unless further change done)
    st.markdown("#### 🔄 Updated Prediction")
    new_pred, new_proba, new_shap = predict_with_explanation(selected_model, whatif_df)
    new_label = "Bad Risk" if new_pred == 1 else "Good Risk"
    st.success(f"🆕 **New Prediction:** `{new_label}` ({new_proba:.2%})")

    # 🔍 SHAP Comparison
    st.markdown("#### 🔍 SHAP Comparison (Original vs What-If)")
    orig_values = orig_shap.values
    new_values = new_shap.values
    feature_names = orig_shap.feature_names
    top_idx = np.argsort(np.abs(orig_values - new_values))[::-1][:5]

    fig, ax = plt.subplots(figsize=(6, 4))
    bar_width = 0.35
    index = np.arange(len(top_idx))

    ax.barh(index, orig_values[top_idx], bar_width, label='Original', color='gray')
    ax.barh(index + bar_width, new_values[top_idx], bar_width, label='What-If', color='orange')
    ax.set_yticks(index + bar_width / 2)
    ax.set_yticklabels([feature_names[i] for i in top_idx])
    ax.invert_yaxis()
    ax.set_xlabel("SHAP Value")
    ax.set_title("Top Changed SHAP Feature Contributions")
    ax.legend()

    st.pyplot(fig)


with st.expander("📊 Top Features Over Time (SHAP-based Logs)"):
    if os.path.exists(log_path):
        df_logs = pd.read_csv(log_path, on_bad_lines='skip')

        if "Top_3_Features" in df_logs.columns and not df_logs["Top_3_Features"].isnull().all():
            # Step 1: One-hot encode each feature in the comma-separated Top_3_Features
            top_feats = df_logs["Top_3_Features"].str.get_dummies(sep=', ')
            
            # Step 2: Sum how many times each feature appears
            top_feats_sum = top_feats.sum().sort_values(ascending=False)

            # Step 3: Plot bar chart
            fig_feat, ax_feat = plt.subplots()
            sns.barplot(x=top_feats_sum.values, y=top_feats_sum.index, ax=ax_feat, palette="viridis")
            ax_feat.set_title("Most Frequently Influential Features")
            ax_feat.set_xlabel("Count in Top 3 Features")
            ax_feat.set_ylabel("Feature Name")
            st.pyplot(fig_feat)

        else:
            st.info("🔍 'Top_3_Features' column is missing or empty in the logs.")
    else:
        st.info("ℹ️ No prediction logs found yet. Run a prediction to generate logs.")