# 📞 Telco Customer Churn Prediction
# Telco Customer Churn Prediction

**End-to-end machine learning project to predict which telecom customers are likely to leave (churn)**

Built as a hands-on data science learning project using the classic Telco Customer Churn dataset (~7,000 customers).

### What this project does
- Loads and cleans real-world telecom data
- Performs exploratory data analysis → discovers strong churn signals (short tenure, month-to-month contracts, fiber optic, electronic check payments, few add-on services)
- Engineers useful features (tenure buckets, total services count, autopay flag, etc.)
- Builds and compares models: Logistic Regression, Random Forest, XGBoost
- Tunes the best model → **Tuned Logistic Regression achieves AUC ≈ 0.863**
- Deploys an interactive web app with **Streamlit** for real-time churn probability predictionset.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
</p>

## ✨ Highlights

- 🔍 **Best model**: Tuned Logistic Regression  
- 🎯 **Performance**: AUC ≈ **0.863**  
- 📊 Interactive **Streamlit web app** for real-time predictions  
- 🛠 Full pipeline: cleaning → EDA → feature engineering → modeling  
- 📈 Strongest signals found: month-to-month contracts, short tenure, fiber optic, electronic check, no add-on services

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR-USERNAME/telco-churn-prediction.git
cd telco-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the demo app
streamlit run app.py

