import joblib
import pandas as pd
import numpy as np
import json
import requests
import time
from flask import Flask, request, jsonify, render_template
from pathlib import Path
app = Flask(__name__)
MODEL_PATH = Path('static') / 'dropout_prediction_model.joblib'
FEATURES_PATH = Path('static') / 'required_features.json'
model = None
REQUIRED_FEATURES = []
FINAL_CATEGORICAL_COLS = [
    'gender', 'department', 'scholarship', 'parental_education', 
    'extra_curricular', 'sports_participation'
]
def load_deployment_artifacts():
    """Loads the model and the required feature list from static files."""
    global model, REQUIRED_FEATURES
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
    try:
        with open(FEATURES_PATH, 'r') as f:
            REQUIRED_FEATURES = json.load(f)
        print(f"Loaded {len(REQUIRED_FEATURES)} required features from JSON.")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load required_features.json from {FEATURES_PATH}. "
              f"Please ensure you ran modeltraining.py successfully to create this file. Error: {e}")
        REQUIRED_FEATURES = []
load_deployment_artifacts()
def preprocess_input(data):
    """
    Converts raw user input (from the frontend form) into a DataFrame 
    format exactly matching the model's REQUIRED_FEATURES.
    """
    input_series = pd.Series(data)
    df = pd.DataFrame([input_series])
    numeric_cols = ['age', 'cgpa', 'attendance_rate', 'family_income', 'past_failures', 
                    'study_hours_per_week', 'assignments_submitted', 'projects_completed', 
                    'total_activities']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    final_features_df = pd.DataFrame(0, index=[0], columns=REQUIRED_FEATURES)
    for col_name in FINAL_CATEGORICAL_COLS:
        input_value = str(df[col_name].iloc[0]).lower().strip()
        if col_name == 'gender':
            if input_value == 'male' and 'gender_male' in REQUIRED_FEATURES:
                final_features_df['gender_male'] = 1
            
        elif col_name == 'department':
            ohe_col = f'department_{input_value.upper()}'
            if ohe_col in REQUIRED_FEATURES:
                final_features_df[ohe_col] = 1
                
        elif col_name == 'scholarship' and input_value == 'yes':
            if 'scholarship_yes' in REQUIRED_FEATURES:
                final_features_df['scholarship_yes'] = 1

        elif col_name == 'extra_curricular' and input_value == 'yes':
            if 'extra_curricular_yes' in REQUIRED_FEATURES:
                final_features_df['extra_curricular_yes'] = 1
                
        elif col_name == 'sports_participation' and input_value == 'yes':
            if 'sports_participation_yes' in REQUIRED_FEATURES:
                final_features_df['sports_participation_yes'] = 1

        elif col_name == 'parental_education':
            ohe_map = {
                'postgraduate': 'parental_education_Postgraduate',
                'primary': 'parental_education_Primary',
                'secondary': 'parental_education_Secondary',
            }
            mapped_col = ohe_map.get(input_value)
            if mapped_col and mapped_col in REQUIRED_FEATURES:
                 final_features_df[mapped_col] = 1
    for col in final_features_df.columns:
        if col in df.columns:
            final_features_df[col] = df[col]
        elif 'AVERAGE' in col:
            if 'attendance_rate' in col and 'attendance_rate' in df.columns:
                 final_features_df[col] = df['attendance_rate']
            elif 'study_hours_per_week' in col and 'study_hours_per_week' in df.columns:
                 final_features_df[col] = df['study_hours_per_week']
    return final_features_df[REQUIRED_FEATURES]

@app.route('/')
def home():
    """Renders the main prediction interface."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to receive student data and return dropout prediction probability.
    """
    if model is None or not REQUIRED_FEATURES:
        return jsonify({'error': 'Model or feature list not loaded. Check server console.'}), 500

    try:
        data = request.json
        processed_data = preprocess_input(data)

        processed_data = processed_data.fillna(0)
        
        
        probability_dropout = model.predict_proba(processed_data)[:, 1][0]
        
        risk_threshold = 0.25 
        
        return jsonify({
            'success': True,
            'probability': float(f"{probability_dropout:.4f}"),
            'prediction_status': 'High Risk' if probability_dropout >= risk_threshold else 'Low Risk',
            'confidence_level': f"{(1 - probability_dropout):.2f}" if probability_dropout < risk_threshold else f"{probability_dropout:.2f}",
            'input_data': data 
        })

    except Exception as e:
        import traceback
        print(f"Prediction Error: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f"An error occurred during prediction: {e}",
            'detail': 'Check server logs for full traceback.'
        }), 500

@app.route('/chat', methods=['POST'])
def chat_interventions():
    """
    API endpoint to provide conversational intervention suggestions using Gemini.
    """
    data = request.json
    risk_status = data.get('risk_status', 'Unknown Risk')
    input_data = data.get('input_data', {})
    data_points = [f"{key.replace('_', ' ').title()}: {value}" for key, value in input_data.items()]
    student_profile = "\n- " + "\n- ".join(data_points)
    
    system_prompt = (
        "You are a compassionate, professional, and highly experienced college academic advisor. "
        "Your goal is to provide specific, actionable, and positive intervention advice to a student's counselor. "
        "The advice must be tailored to the student's risk status and profile data. "
        "Keep the response concise (2-3 paragraphs max) and focus on 2-3 key action points."
    )
    
    user_query = (
        f"A student has been flagged with a **{risk_status}** risk of dropout based on their profile data. "
        "The profile details are as follows:\n\n"
        f"{student_profile}\n\n"
        "Please provide the counselor with specific, evidence-based intervention steps and guidance to support this student and mitigate their risk. "
        "Start your response with a clear summary of the student's primary area of concern based on the provided data."
    )
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "tools": [{"google_search": {}}], 
    }
    
    API_KEY = "AIzaSyDoZqItuWvZ8XKMwneAW8L_f2CKQ-dy_3Y" 
    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"
    for attempt in range(3):
        try:
            response = requests.post(API_URL, json=payload, headers={'Content-Type': 'application/json'})
            response.raise_for_status() 
            
            result = response.json()
            generated_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'Could not generate advice.')
            
            return jsonify({
                'success': True,
                'advice': generated_text
            })

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and attempt < 2:
                wait_time = 2 ** attempt
                print(f"Rate limit hit. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"API call failed on attempt {attempt}: {e}")
                return jsonify({'success': False, 'error': f"API call failed: {e}", 'detail': response.text}), 500
        except Exception as e:
             return jsonify({'success': False, 'error': f"An unexpected error occurred: {e}"}), 500


if __name__ == '__main__':
    print("\n*******************************************************************")
    print("Application Ready! Please open the following URL in your browser:")
    print("   ---> http://127.0.0.1:5000/")
    print("*******************************************************************\n")
    app.run(debug=True)
