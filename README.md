# 📊 Direct Marketing Optimization

## 🎯 Objective
The goal of this project is to optimize direct marketing campaigns by predicting purchase propensity and expected revenue, then selecting high-value clients under given constraints.

---

## 📌 Problem Statement
A bank is running a direct marketing campaign for three products:
- **Consumer Loans** (CL)
- **Credit Cards** (CC)
- **Mutual Funds** (MF)

The bank can contact **only 15% of clients**, and each client can receive **only one offer**. The goal is to **maximize total revenue** by targeting high-propensity clients.

---

## 📂 Data Overview
The dataset consists of the following:
- **Client Information**: Age, gender, bank tenure.
- **Product Holdings**: Account types, balances, and transaction history.
- **Sales & Revenue**:
  - `Sale_MF`, `Sale_CC`, `Sale_CL` → Binary purchase indicators (1 = Purchased, 0 = Not Purchased)
  - `Revenue_MF`, `Revenue_CC`, `Revenue_CL` → Revenue generated from a purchase (available for 60% of clients)

---

## 🏗️ Approach
### **1. Data Preprocessing**
- Clean missing values and preprocess categorical & numerical features.
- Standardize numerical features for better model performance.

### **2. Predict Purchase Propensity**
- Train **three Logistic Regression models** to predict the probability of a client purchasing **each product** (`Sale_*`).
- Evaluate models using **AUROC**.

### **3. Predict Expected Revenue**
- Train **three Linear Regression models** to predict revenue (`Revenue_*`).
- Revenue models are trained **only on clients who have purchased the product**.

### **4. Optimize Target Selection**
- Compute **expected revenue**:  
  $$\text{Expected Revenue} = P(\text{purchase}) \times \text{Predicted Revenue}$$
- Rank clients by **highest expected revenue**.
- Select **top 15%** of clients and assign them the offer with the **highest expected revenue**.

---

## 🚀 How to Run the Project

### **1️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2️⃣ Run the Data Processing & Modeling Pipeline**
```bash
python train_models.py
```

### **3️⃣ Generate the Final Target List**
```bash
python select_clients.py
```

### **4️⃣ View Results**
Check the `results/selected_clients.csv` file for the final targeted client list.

---

## 📈 Evaluation Metrics
- **Propensity Model Performance**: AUROC Score
- **Revenue Model Performance**: RMSE
- **Marketing Optimization**: Total expected revenue

---

## 🛠️ Tech Stack
- **Programming Language**: Python 🐍
- **Libraries**: `pandas`, `scikit-learn`, `numpy`, `matplotlib`
- **Version Control**: Git & GitHub
---

## 💡 Future Improvements
- Use **XGBoost** or **Neural Networks** for better revenue prediction.
- Implement **multi-objective optimization** to balance revenue and client fairness.
- Deploy as an AWS sagemaker endpoint for real-time scoring.
- Add pytests for prodcuction unit code testing for any merges. 
  
---

### 🌟 If you find this project useful, give it a ⭐ on GitHub!

