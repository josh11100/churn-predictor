import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap

from src.preprocess import get_train_test


def train_logistic(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_scaled, y_train)
    return model, scaler


def train_xgboost(X_train, y_train):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, scaler=None, label="Model"):
    X = scaler.transform(X_test) if scaler else X_test
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    print(f"\n{'='*40}")
    print(f"  {label}")
    print(f"{'='*40}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    return y_proba


def plot_shap(model, X_train, feature_names):
    print("\nGenerating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    plt.figure()
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
    print("SHAP plot saved to shap_summary.png")
    plt.close()


def main():
    X_train, X_test, y_train, y_test, features = get_train_test("data/telco_churn.csv")

    print("Training Logistic Regression (baseline)...")
    lr_model, scaler = train_logistic(X_train, y_train)
    evaluate(lr_model, X_test, y_test, scaler=scaler, label="Logistic Regression")

    print("\nTraining XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    evaluate(xgb_model, X_test, y_test, label="XGBoost")

    # Save the best model
    joblib.dump({"model": xgb_model, "features": features}, "model.pkl")
    print("\n✅ Model saved to model.pkl")

    # SHAP explainability
    plot_shap(xgb_model, X_train, features)


if __name__ == "__main__":
    main()
