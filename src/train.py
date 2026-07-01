import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import cross_val_score
from src.utils import setup_directories, get_logger
from src.preprocess import preprocess_train_data
from src.evaluate import calculate_metrics, generate_evaluation_plots
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

logger = get_logger("TrainEngine")

def main():
    setup_directories()
    dataset_path = 'data/fitness_dataset.csv'
    
    if not os.path.exists(dataset_path) and os.path.exists('fitness_dataset.csv'):
        import shutil
        shutil.copy('fitness_dataset.csv', dataset_path)
        
    X_train, X_test, y_train, y_test = preprocess_train_data(dataset_path)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=6),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Support Vector Machine': SVC(random_state=42, probability=True)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        results[name] = calculate_metrics(y_test, preds, probs)
        
    best_name = max(results, key=lambda k: results[k]['F1-Score'])
    joblib.dump(models[best_name], 'models/best_model.pkl')
    
    with open('models/winner_meta.txt', 'w') as f:
        f.write(best_name)
        
    generate_evaluation_plots(best_name, models[best_name], X_test, y_test, results)
    print(f"Workflow Complete! Champion Model: {best_name}")

if __name__ == '__main__':
    main()