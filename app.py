import streamlit as st
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
            st.error(f"Result: NOT FIT (Confidence: {prob:.2%})")