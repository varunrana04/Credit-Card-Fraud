# Credit Card Fraud Detection (Ensemble + SHAP Explainability)

## 📌 Overview
This project focuses on detecting **fraudulent credit card transactions** using machine learning.  
The solution applies **ensemble methods** (bagging + boosting) and provides **model explainability with SHAP** — including **feature interactions** for interpretability.  
Given the high class imbalance in the dataset, specialized techniques for **resampling and evaluation** were applied.

---

## 🛠️ Tech Stack
- **Python 3.9+**
- **Scikit-learn** (Ensemble models, metrics)
- **XGBoost / LightGBM**
- **SHAP** for explainability
- **NumPy, Pandas, Seaborn, Matplotlib**
- **Kaggle** for experimentation

---

## 📊 Dataset
- Source: [Credit Card Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- Transactions: **284,807**  
- Features: **30** (anonymized principal components + `Amount`, `Time`)  
- Labels: `0 = Non-Fraud`, `1 = Fraud`  
- Fraudulent transactions: **0.172%** (highly imbalanced dataset)  

---

## 🔍 Methodology

1. **EDA**
   - Distribution of fraudulent vs. non-fraudulent transactions
   - Correlation analysis across anonymized features
   - Boxplots of `Amount` and `Time`

2. **Preprocessing**
   - Scaling with StandardScaler
   - Handling class imbalance with **undersampling / SMOTE**
   - Splitting into train/test sets (stratified sampling)

3. **Modeling**
   - Ensemble approaches tested:
     - **Random Forest**
     - **XGBoost**
     - **LightGBM**
   - Stacked ensemble for final predictions

4. **Explainability**
   - **SHAP values** to explain global + local predictions
   - **SHAP interaction plots** for feature dependency analysis

---

## 📈 Results

### Classification Metrics
| Model         | Precision | Recall | F1-Score | AUC-ROC |
|---------------|-----------|--------|----------|---------|
| Random Forest | XX%       | XX%    | XX%      | XX%     |
| XGBoost       | XX%       | XX%    | XX%      | XX%     |
| LightGBM      | XX%       | XX%    | XX%      | XX%     |
| **Ensemble**  | **XX%**   | **XX%**| **XX%**  | **XX%** |

*(I Have to Re-Run the whole program, if anyone need the stats of the project do message me)*

### SHAP Explainability
- Top influential features: `V14`, `V12`, `V17`  
- Fraudulent cases showed strong **negative SHAP contributions** from V14 & V17  
- SHAP summary + dependence plots provided  

---

## 🚀 Features
- Handles **imbalanced fraud detection problem**  
- Combines **ensemble learning** for robustness  
- **Full SHAP explainability**, including **feature interaction analysis**  
- End-to-end pipeline from **EDA → Preprocessing → Modeling → Explainability**

---

## 📂 Project Structure
Credit-Card-Fraud-Detection/
│── data/
│── eda_notebook.ipynb
│── preprocessing.ipynb
│── models/
│ ├── random_forest.pkl
│ ├── xgboost.pkl
│ ├── ensemble.pkl
│── shap_analysis.ipynb
│── README.md

## ▶️ Usage

### Train models
```bash
python train.py
python shap_analysis.py
