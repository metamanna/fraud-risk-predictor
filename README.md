# fraud-risk-predictor
A Streamlit app that predicts loan applicant creditworthiness using machine learning, explainable AI (SHAP, LIME), sentiment analysis, and PDF reporting.
https://fraud-risk-predictor-tamannaaggarwal.streamlit.app/

# 🧠 Creditworthiness Prediction Dashboard

A Streamlit web app that predicts whether a loan applicant is a **Good or Bad Credit Risk** using machine learning and explainable AI. The app also integrates **SHAP**, **LIME**, **Transformer-based sentiment analysis**, and generates a downloadable **PDF report** for each prediction.

---

## 🚀 Features

- ✅ Predict credit risk (Good / Bad) using trained ML models (Random Forest, CatBoost, XGBoost)
- 📊 SHAP and LIME explanations for model interpretability
- 💬 Justification-based sentiment analysis using Transformer models
- 📄 Automatic PDF report generation with inputs, results, and explanations
- 🔁 What-If Scenario Tool to simulate changes in input features
- 📈 Model comparison (AUC, classification report, confusion matrix)
- ⚠️ Input validation (detects gibberish justifications)

---

## 🧠 Technologies Used

- **Python**, **Streamlit**, **Pandas**, **Scikit-learn**
- **SHAP**, **LIME**, **FPDF**
- **Transformers** (`bert-base-uncased`, Sentence Transformers)
- **CatBoost**, **XGBoost**
- **SMOTE** (Synthetic Oversampling)

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/credit-risk-predictor.git
cd credit-risk-predictor
pip install -r requirements.txt
streamlit run main.py

Example Inputs
Duration: 24

Credit Amount: 5000

Age: 35

Purpose: radio/tv

Justification: "I need to renovate my home and cover utility expenses"

📄 PDF Report
After prediction, click "Explain Prediction" → a downloadable PDF will be generated with:

Model prediction

Sentiment analysis

SHAP feature impact

User input summary

🔒 Future Improvements
Add user login/authentication

Deploy with Hugging Face Spaces

Add model versioning and A/B testing

👤 Author
Made with ❤️ by Tamanna Aggarwal
📧 tamanna08aggarwal@gmail.com
🔗 www.linkedin.com/in/tamanna-aggarwal-tech
