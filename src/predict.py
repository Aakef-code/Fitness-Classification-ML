import os
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
        return {'is_fit': int(pred), 'confidence': float(prob)}