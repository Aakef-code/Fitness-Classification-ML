import pandas as pd
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
    
    return X_train_final, X_test_final, y_train, y_test