import os

# Define the file structure and contents
files = {
    # 1. Requirements
    "requirements.txt": """pandas>=1.5.0
numpy>=1.22.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
joblib>=1.1.0
streamlit>=1.12.0
ipykernel""",

    # 2. License
    "LICENSE": """MIT License
Copyright (c) 2026
Permission is hereby granted, free of charge, to any person obtaining a copy...""",

    # 3. Git Ignore
    ".gitignore": """__pycache__/
*.py[cod]
*$py.class
.env
.venv
env/
venv/
.ipynb_checkpoints/
models/*.pkl
outputs/*.png
outputs/*.txt
.DS_Store
Thumbs.db""",

    # 4. Utils
    "src/utils.py": """import os
import sys
import logging

def setup_directories():
    directories = ['data', 'src', 'notebooks', 'outputs', 'models']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"[INFO] Created directory: {directory}")

def get_logger(name="FitnessML"):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(name)""",

    # 5. Preprocess
    "src/preprocess.py": """import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os
from src.utils import get_logger

logger = get_logger("Preprocessing")

def clean_smokes_column(df):
    if 'smokes' in df.columns:
        df = df.copy()
        df['smokes'] = df['smokes'].astype(str).str.strip().str.lower()
        mapping = {'no': 0.0, '0': 0.0, '0.0': 0.0, 'yes': 1.0, '1': 1.0, '1.0': 1.0, 'nan': np.nan}
        df['smokes'] = df['smokes'].map(mapping)
    return df

def clean_gender_column(df):
    if 'gender' in df.columns:
        df = df.copy()
        df['gender'] = df['gender'].astype(str).str.strip().str.upper().map({'F': 0, 'M': 1})
    return df

def detect_columns(df, target_col='is_fit'):
    all_features = [col for col in df.columns if col != target_col]
    numerical_cols = [col for col in all_features if df[col].dtype in ['int64', 'float64']]
    categorical_cols = [col for col in all_features if df[col].dtype not in ['int64', 'float64']]
    return numerical_cols, categorical_cols

def preprocess_train_data(filepath, target_col='is_fit', test_size=0.2, random_state=42):
    df = pd.read_csv(filepath)
    df = clean_smokes_column(df)
    df = clean_gender_column(df)
    num_cols, cat_cols = detect_columns(df, target_col)
    
    X = df.drop(columns=[target_col])
    y = df[target_col].fillna(df[target_col].mode()[0])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    num_imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train_num = num_imputer.fit_transform(X_train[num_cols])
    X_test_num = num_imputer.transform(X_test[num_cols])
    
    X_train_scaled = scaler.fit_transform(X_train_num)
    X_test_scaled = scaler.transform(X_test_num)
    
    X_train_final = pd.DataFrame(X_train_scaled, columns=num_cols, index=X_train.index)
    X_test_final = pd.DataFrame(X_test_scaled, columns=num_cols, index=X_test.index)
    
    joblib.dump(num_imputer, 'models/num_imputer.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(num_cols, 'models/numerical_cols_list.pkl')
    joblib.dump(cat_cols, 'models/categorical_cols_list.pkl')
    joblib.dump(X_train_final.columns.tolist(), 'models/feature_columns.pkl')
    
    return X_train_final, X_test_final, y_train, y_test""",

    # 6. Evaluate
    "src/evaluate.py": """import numpy as np
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
    plt.close()""",

    # 7. Train
    "src/train.py": """import pandas as pd
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
    main()""",

    # 8. Predict
    "src/predict.py": """import os
import pandas as pd
import joblib

class FitnessPredictor:
    def __init__(self, models_dir='models'):
        self.model = joblib.load(os.path.join(models_dir, 'best_model.pkl'))
        self.scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        self.imputer = joblib.load(os.path.join(models_dir, 'num_imputer.pkl'))
        self.feature_columns = joblib.load(os.path.join(models_dir, 'feature_columns.pkl'))
        self.numerical_cols = joblib.load(os.path.join(models_dir, 'numerical_cols_list.pkl'))

    def predict_single(self, input_dict):
        df = pd.DataFrame([input_dict])
        df['smokes'] = 1.0 if str(df.get('smokes', [0])[0]).lower() in ['yes', '1'] else 0.0
        df['gender'] = 1.0 if str(df.get('gender', [0])[0]).upper() in ['M', 'MALE'] else 0.0
        
        imputed = self.imputer.transform(df[self.numerical_cols])
        scaled = self.scaler.transform(imputed)
        final_df = pd.DataFrame(scaled, columns=self.numerical_cols)[self.feature_columns]
        
        pred = self.model.predict(final_df)[0]
        prob = self.model.predict_proba(final_df)[0][pred] if hasattr(self.model, "predict_proba") else 1.0
        return {'is_fit': int(pred), 'confidence': float(prob)}""",

    # 9. Streamlit App
    "app.py": """import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Fitness Predictor", page_icon="🏋️‍♂️")
st.title("🏋️‍♂️ Classic ML Fitness Predictor")

@st.cache_resource
def load_pipeline():
    try:
        return {
            'model': joblib.load('models/best_model.pkl'),
            'scaler': joblib.load('models/scaler.pkl'),
            'imputer': joblib.load('models/num_imputer.pkl'),
            'feature_cols': joblib.load('models/feature_columns.pkl'),
            'num_cols': joblib.load('models/numerical_cols_list.pkl')
        }
    except Exception as e:
        return None

pipeline = load_pipeline()

if not pipeline:
    st.error("Model files not found! Please run 'python src/train.py' first.")
else:
    age = st.slider("Age", 18, 90, 30)
    height = st.slider("Height (cm)", 120, 220, 170)
    weight = st.slider("Weight (kg)", 40, 150, 70)
    hr = st.number_value = st.number_input("Heart Rate", value=72.0)
    bp = st.number_input("Blood Pressure (systolic)", value=120.0)
    sleep = st.slider("Sleep (Hours)", 4.0, 12.0, 8.0)
    nutrition = st.slider("Nutrition Score (1-10)", 1.0, 10.0, 6.0)
    activity = st.slider("Activity Index (1-5)", 1.0, 5.0, 3.0)
    smokes = st.selectbox("Smokes?", ["no", "yes"])
    gender = st.selectbox("Gender", ["F", "M"])

    if st.button("Predict Fitness"):
        input_data = pd.DataFrame([[age, height, weight, hr, bp, sleep, nutrition, activity, 
                                    1.0 if smokes == "yes" else 0.0, 1.0 if gender == "M" else 0.0]], 
                                  columns=pipeline['num_cols'])
        
        imputed = pipeline['imputer'].transform(input_data)
        scaled = pipeline['scaler'].transform(imputed)
        final_df = pd.DataFrame(scaled, columns=pipeline['num_cols'])[pipeline['feature_cols']]
        
        pred = pipeline['model'].predict(final_df)[0]
        prob = pipeline['model'].predict_proba(final_df)[0][pred] if hasattr(pipeline['model'], "predict_proba") else 1.0
        
        if pred == 1:
            st.success(f"Result: FIT (Confidence: {prob:.2%})")
        else:
            st.error(f"Result: NOT FIT (Confidence: {prob:.2%})")"""
}

# Create directories and files
for filepath, content in files.items():
    # Create folders if they do not exist
    dir_name = os.path.dirname(filepath)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    # Write the file contents
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {filepath}")

print("\\nSuccess! All ML project folders and code files generated.")