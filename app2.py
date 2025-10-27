import joblib
import pandas as pd
import numpy as np
import io
from flask import Flask, request, jsonify, render_template
from pathlib import Path

app = Flask(__name__)
MODEL_PATH = Path('static') / 'dropout_prediction_model.joblib'
CLEANED_DATA_PATH = Path('cleaned_and_encoded_dataset.csv')

model = None
REQUIRED_FEATURES = []
def initialize_model_and_features():
    """Loads the model and extracts the exact feature names from the cleaned dataset."""
    global model, REQUIRED_FEATURES
    
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        return

    try:
        csv_content = __content_fetcher__.get_content(CLEANED_DATA_PATH.name)
        df_reference = pd.read_csv(io.StringIO(csv_content), index_col=None)
        REQUIRED_FEATURES = [
            'age', 'cgpa', 'attendance_rate', 'family_income', 'past_failures',
            'study_hours_per_week', 'assignments_submitted', 'projects_completed',
            'total_activities', 'Unnamed: 17', 'AVERAGE of study_hours_per_week', 
            'AVERAGE of attendance_rate', 'gender_male', 'gender_other', 
            'department_BIO', 'department_CIVIL', 'department_COMMERCE', 
            'department_CS', 'department_ECE', 'department_ME', 'scholarship_yes', 
            'parental_education_Postgraduate', 'parental_education_Primary', 
            'parental_education_Secondary', 'extra_curricular_yes', 
            'sports_participation_yes', 'department.1_ARTS', 'department.1_ARTS Total', 
            'department.1_BIO', 'department.1_BIO Total', 'department.1_CIVIL', 
            'department.1_CIVIL Total', 'department.1_COMMERCE', 'department.1_COMMERCE Total', 
            'department.1_CS', 'department.1_CS Total', 'department.1_ECE', 
            'department.1_ECE Total', 'department.1_Grand Total', 'department.1_ME', 
            'department.1_ME Total', 'department.1_nan Total'
        ]
        if model is not None and hasattr(model, 'n_features_in_') and model.n_features_in_ != len(REQUIRED_FEATURES):
            print(f"Warning: Model expects {model.n_features_in_} features, but extracted list has {len(REQUIRED_FEATURES)} features.")
            print("The model may have been trained on a subset of the dataset's columns.")
        
        print(f"Features loaded: {len(REQUIRED_FEATURES)} columns.")
        
    except Exception as e:
        print(f"Error loading required features from CSV: {e}")
        REQUIRED_FEATURES = [
            'age', 'cgpa', 'attendance_rate', 'family_income', 'past_failures',
            'study_hours_per_week', 'assignments_submitted', 'projects_completed',
            'total_activities', 'gender_male', 'gender_other',
            'department_BIO', 'department_CIVIL', 'department_COMMERCE', 'department_CS',
            'department_ECE', 'department_ME',
            'scholarship_yes',
            'parental_education_Postgraduate', 'parental_education_Primary', 'parental_education_Secondary',
            'extra_curricular_yes',
            'sports_participation_yes'
        ]
        print("Using a simplified, default feature list due to error reading CSV.")
def preprocess_input(data):
    """
    Converts raw user input into a DataFrame format suitable for the model,
    generating ALL required features (including the strange ones) and setting them to zero.
    """
    if not REQUIRED_FEATURES:
        raise ValueError("Required features list is empty. Initialization failed.")
    input_series = pd.Series(data)
    df = pd.DataFrame([input_series])
    df['gender'] = df['gender'].str.lower().str.strip().replace({'m': 'male', 'f': 'female', 'other': 'other'})
    df['department'] = df['department'].str.upper().str.strip()
    df['scholarship'] = df['scholarship'].str.lower().str.strip()
    df['extra_curricular'] = df['extra_curricular'].str.lower().str.strip()
    df['sports_participation'] = df['sports_participation'].str.lower().str.strip()
    final_features_df = pd.DataFrame(0.0, index=df.index, columns=REQUIRED_FEATURES)
    for col in ['age', 'cgpa', 'attendance_rate', 'family_income', 'past_failures',
                'study_hours_per_week', 'assignments_submitted', 'projects_completed',
                'total_activities']:
        if col in final_features_df.columns:
            final_features_df[col] = df[col].astype(float).iloc[0]
    if 'AVERAGE of study_hours_per_week' in final_features_df.columns:
        final_features_df['AVERAGE of study_hours_per_week'] = 0.0
    if 'AVERAGE of attendance_rate' in final_features_df.columns:
        final_features_df['AVERAGE of attendance_rate'] = 0.0
    if df['gender'].iloc[0] == 'male':
        final_features_df['gender_male'] = 1
    elif df['gender'].iloc[0] == 'other':
        final_features_df['gender_other'] = 1
    department = df['department'].iloc[0]
    if department == 'BIO': final_features_df['department_BIO'] = 1
    elif department == 'CIVIL': final_features_df['department_CIVIL'] = 1
    elif department == 'COMMERCE': final_features_df['department_COMMERCE'] = 1
    elif department == 'CS': final_features_df['department_CS'] = 1
    elif department == 'ECE': final_features_df['department_ECE'] = 1
    elif department == 'ME': final_features_df['department_ME'] = 1
    if df['scholarship'].iloc[0] == 'yes':
        final_features_df['scholarship_yes'] = 1
    parental_edu = df['parental_education'].iloc[0]
    if parental_edu == 'Postgraduate': final_features_df['parental_education_Postgraduate'] = 1
    elif parental_edu == 'Primary': final_features_df['parental_education_Primary'] = 1
    elif parental_edu == 'Secondary': final_features_df['parental_education_Secondary'] = 1
    if df['extra_curricular'].iloc[0] == 'yes':
        final_features_df['extra_curricular_yes'] = 1
    if df['sports_participation'].iloc[0] == 'yes':
        final_features_df['sports_participation_yes'] = 1
    return final_features_df[REQUIRED_FEATURES]

@app.route('/')
def home():
    """Renders the main prediction interface."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to receive student data and return dropout prediction probability."""
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500
    if not REQUIRED_FEATURES:
        return jsonify({'error': 'Feature list not initialized. Check server logs.'}), 500

    try:
        data = request.json
        processed_data = preprocess_input(data)
        probability_dropout = model.predict_proba(processed_data)[:, 1][0]
        tuned_threshold = 0.26 

        return jsonify({
            'success': True,
            'probability': float(f"{probability_dropout:.4f}"),
            'prediction_status': 'High Risk' if probability_dropout >= tuned_threshold else 'Low Risk',
            'threshold_used': tuned_threshold
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        if 'Feature names should match' in str(e):
             return jsonify({
                'error': 'Feature Mismatch Error.',
                'details': 'The number or name of input features did not match the model\'s required features. This is often due to missing preprocessing steps.',
                'exception': str(e)
            }), 500
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    initialize_model_and_features()
    app.run(debug=True)
