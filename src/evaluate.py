import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, classification_report
import os

def calculate_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0)
    }
    metrics['ROC-AUC'] = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.5
    return metrics

def generate_evaluation_plots(best_model_name, best_model, X_test, y_test, results_dict):
    os.makedirs('outputs', exist_ok=True)
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
    
    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.savefig('outputs/confusion_matrix.png', dpi=300)
    plt.close()
    
    # Model Comparison
    plt.figure(figsize=(10, 6))
    pd.DataFrame(results_dict).T[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(kind='bar')
    plt.tight_layout()
    plt.savefig('outputs/model_comparison.png', dpi=300)
    plt.close()