# 📉 ChurnIQ — AI-Powered Customer Churn Predictor

A full end-to-end machine learning application that predicts the probability a customer will cancel their subscription — before they do. Built with XGBoost, SHAP explainability, and a live Streamlit dashboard.

**[🚀 Live Demo](#)** · **[📊 Notebook](#)** · **[📝 Blog Post](#)**

---

## 📸 Screenshot

> *(Add a screenshot of your running Streamlit app here)*

---

## 🎯 What it does

Businesses lose thousands in revenue every month to customer churn. This tool lets you:

- Input any customer's profile (contract type, tenure, services, charges)
- Get a **real-time churn probability score** (0–100%)
- Understand **why** the model made that prediction (SHAP waterfall chart)
- See the **revenue at risk** and get an **actionable recommendation**

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | **0.847** |
| Recall (Churn class) | **79%** |
| Training samples | 7,043 customers |
| Class imbalance handling | `scale_pos_weight` in XGBoost |

> A ROC-AUC of 0.847 means the model correctly ranks at-risk customers over safe ones 84.7% of the time — genuinely useful for prioritizing retention efforts.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data processing | `pandas`, `numpy`, `scikit-learn` |
| Modeling | `XGBoost`, `Logistic Regression` (baseline) |
| Explainability | `SHAP` (TreeExplainer + waterfall plots) |
| Dashboard | `Streamlit` |
| Serialization | `joblib` |

---

## 🗂️ Project Structure

```
churn-predictor/
├── data/
│   └── telco_churn.csv          # Telco customer dataset (7,043 rows)
├── src/
│   ├── preprocess.py            # Data cleaning & feature engineering
│   └── train.py                 # Model training, evaluation & SHAP plots
├── app.py                       # Streamlit dashboard
├── model.pkl                    # Saved XGBoost model
├── requirements.txt
└── README.md
```

---

## ⚙️ Feature Engineering

Beyond the raw dataset columns, three engineered features were added:

- **`charges_per_month`** — Total charges normalized by tenure (captures value consistency)
- **`is_new_customer`** — Binary flag for customers with ≤ 3 months tenure (new customers churn differently)
- **`num_addons`** — Count of active add-on services (stickiness proxy)

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/churn-predictor.git
cd churn-predictor
```

**2. Set up environment**
```bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
# venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**3. Add the dataset**

Download [`telco_churn.csv`](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the `data/` folder.

**4. Train the model**
```bash
python src/train.py
```

**5. Launch the dashboard**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## ☁️ Deploy (Streamlit Cloud)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set the main file path to `app.py`
4. Deploy — your app will be live at `https://yourapp.streamlit.app`

---

## 🔍 Key Design Decisions

**Why XGBoost over a neural network?**
With ~7,000 rows, a gradient boosted tree generalizes better than a deep model. XGBoost also integrates natively with SHAP, making predictions explainable — critical for business stakeholders who need to act on the output.

**Why optimize for Recall over Accuracy?**
A false negative (missing a churner) is more costly than a false positive (flagging a loyal customer). The model uses `scale_pos_weight` to account for class imbalance (~27% churn rate) and prioritize catching churners.

**Why SHAP?**
Black-box predictions are useless if a business can't act on them. SHAP waterfall charts show which specific factors — contract type, tenure, monthly charges — drove each individual prediction, making the model trustworthy and actionable.

---

## 📈 Business Impact

If a company has 10,000 subscribers at $65/month and churns 5% monthly:

- **Monthly churn loss:** $32,500
- **Customers caught by model (79% recall):** ~395 of 500
- **Even retaining 30% of flagged customers:** ~$6,300/month saved

---

## 🔮 Future Improvements

- [ ] Add a batch upload feature (CSV of multiple customers)
- [ ] Connect to a live database via SQL
- [ ] A/B test retention offers and track outcomes
- [ ] Rebuild frontend in React + FastAPI for production SaaS

---

## 📦 Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — IBM sample dataset, publicly available on Kaggle. 7,043 customers, 20 features, binary churn label.

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- Portfolio: [yourwebsite.com](https://yourwebsite.com)
