import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os 

# --- Configuration ---
RAW_DATA_FILE = "Dataset.csv"
ENCODED_DATA_FILE = "cleaned_and_encoded_dataset.csv"

# Function to safely read a file, supporting both local and Canvas environments
def safe_read_csv(filename):
    """Reads a CSV file, checking if the Canvas content fetcher is available first."""
    if '__content_fetcher__' in globals() and hasattr(__content_fetcher__, 'get_content'):
        # Running in the Canvas environment
        try:
            content = __content_fetcher__.get_content(filename)
            return pd.read_csv(io.StringIO(content), index_col=None)
        except Exception as e:
            print(f"Content fetcher error for {filename}: {e}")
            raise FileNotFoundError(f"Could not read file using content fetcher: {filename}")
    else:
        # Running locally
        return pd.read_csv(filename, index_col=None)

# --- Data Loading and Cleaning ---
try:
    df = safe_read_csv(RAW_DATA_FILE)
except FileNotFoundError:
    print(f"Error: Raw data file '{RAW_DATA_FILE}' not found. Please ensure it is in the same directory.")
    raise

# Cleaning steps from your original script (Ensuring proper handling of categorical data)
df['gender'] = df['gender'].astype(str).str.lower().str.strip().replace({'m': 'male', 'f': 'female', 'nan': pd.NA})
df['scholarship'] = df['scholarship'].astype(str).str.lower().str.strip().replace({'y': 'yes', 'n': 'no', 'nan': pd.NA, 'nope': 'no'})
df['extra_curricular'] = df['extra_curricular'].astype(str).str.lower().str.strip().replace({'-': 'no', 'nan': pd.NA, 'n': 'no', 'y': 'yes'})
df['sports_participation'] = df['sports_participation'].astype(str).str.lower().str.strip().replace({'y': 'yes', 'nan': pd.NA, 'n': 'no'})
df['family_income'] = df['family_income'].replace(0, np.nan)
df.loc[df['family_income'] < 0, 'family_income'] = np.nan
df_model = df.drop(columns=['student_id'])
target = 'dropout'

# --- CRITICAL STEP: Load and Prepare Encoded Features ---
try:
    df_encoded = safe_read_csv(ENCODED_DATA_FILE)
    
    # 1. Define target
    y = df_encoded['dropout'].fillna(0).astype(int) 
    
    # 2. Define features (X) by dropping all target columns and problematic summary columns
    X = df_encoded.drop(columns=['dropout', 'dropout.1'], errors='ignore')
    
    # CRITICAL FIX: Drop columns containing ' Total', 'Unnamed', or 'nan Total' that caused the ValueError
    columns_to_drop = [
        col for col in X.columns 
        if ' Total' in col or 'Unnamed:' in col or 'nan Total' in col
    ]
    X = X.drop(columns=columns_to_drop, errors='ignore')
    
    # Final list of clean feature columns
    X_columns = X.columns.tolist()
    
    # Ensure all remaining columns are numerical (convert True/False to 1/0)
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
        elif X[col].dtype == 'object':
            # Attempt to convert objects to numeric, setting errors='coerce' to NaN
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Handle NaNs created during coercion or present from original data (Simple Imputation)
    X = X.fillna(X.mean())
    
except Exception as e:
    print(f"Error loading or cleaning encoded features from '{ENCODED_DATA_FILE}': {e}")
    raise RuntimeError(f"Cannot proceed with training: {e}")

# --- Model Training ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    random_state=42, 
    class_weight='balanced'
)
model.fit(X_train, y_train)

# --- Performance Tuning and Metrics ---
y_proba = model.predict_proba(X_test)[:, 1]
target_recall = 0.8
best_threshold = 0.0
best_recall = 0.0

# Find best threshold for target recall (0.8)
for threshold in np.arange(0.01, 0.51, 0.001):
    y_pred_new = (y_proba >= threshold).astype(int)
    report = classification_report(y_test, y_pred_new, output_dict=True, zero_division=0)
    current_recall = report['1']['recall']
    
    if current_recall >= target_recall:
        best_threshold = threshold
        best_recall = current_recall
        break

    if current_recall > best_recall:
         best_recall = current_recall
         best_threshold = threshold
         
y_pred_tuned = (y_proba >= best_threshold).astype(int)

# --- Save Deployment Artifacts ---
os.makedirs('static', exist_ok=True) 

joblib.dump(model, 'static/dropout_prediction_model.joblib')

# Save the exact feature names used for training (CRITICAL for app.py)
with open('static/required_features.json', 'w') as f:
    json.dump(X_columns, f)
    
print("\n--- Deployment Artifacts Saved ---")
print("1. static/dropout_prediction_model.joblib (Model)")
print("2. static/required_features.json (Exact feature list for Flask)")
print(f"Total features saved: {len(X_columns)}")
print("----------------------------------")

print("---Model Performance (Random Forest with Tuned Threshold)---")
if best_recall < target_recall:
    print(f"**WARNING: Target Recall of {target_recall*100:.2f}% NOT achieved.**")
    print(f"Max Recall achieved: {best_recall*100:.2f}% at Threshold: {best_threshold:.2f}")

print(f"Threshold used: {best_threshold:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_tuned, zero_division=0))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# Feature Importance Plot
feature_importances = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_importances.index, palette="viridis")
plt.title('Top 10 Feature Importance for Dropout Prediction')
plt.xlabel('Feature Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('static/feature_importance.png')
print("Feature importance chart saved to static/feature_importance.png")
