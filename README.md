# Student Dropout Risk Predictor

Submission for: Omnitrics Hackathon 2025

Short Project Summary

This project implements an Intelligent Student Dropout Risk Prediction System designed to identify at-risk students proactively. It utilizes a machine learning model (Random Forest Classifier) trained on various student metrics (CGPA, attendance, study hours, etc.) to predict the probability of dropout.

The solution is deployed as a single-page web application using Flask and includes a key Bonus Feature: a Conversational Chatbot Widget powered by the Gemini API that provides personalized, actionable intervention suggestions based on the predicted risk level.

Tools / Technologies Used

Backend Framework: Flask (Python)

Machine Learning: Scikit-learn, Random Forest Classifier

Data Handling: Pandas, NumPy, Joblib (for model serialization)

Conversational AI: Google Gemini API (via Python requests)

Frontend: HTML5, Tailwind CSS (for responsive design), JavaScript (for API integration)

Instructions to Run Your Project

This project requires Python 3.9+ and the following packages.

## 1. Setup & Environment

Clone or Download this project repository.

Navigate to the project directory in your terminal.

(Recommended) Create and activate a virtual environment:

python -m venv .venv
## On Windows
.venv\Scripts\activate
## On macOS/Linux
source .venv/bin/activate


## 2. Install Dependencies

Install all required Python packages:

pip install -r requirements.txt


## 3. API Key Configuration (Crucial Step)

The Chatbot (Intervention Advisor) requires a Gemini API Key.

Obtain your key from Google AI.

Open the app.py file.

Replace the placeholder with your actual key in the MY_API_KEY variable:

MY_API_KEY = "YOUR_GEMINI_API_KEY_HERE"  # <-- Paste your actual key here


## 4. Run the Application

Start the Flask server:

python app.py


## 5. Access the Dashboard

Open your web browser.

Navigate to the local host address provided in the terminal (usually: http://127.0.0.1:5000/).

## How to Test:

Fill out all fields in the Student Profile Simulation form.

Click "Predict Dropout Risk".

View the Risk Status (High/Low).

Click the floating Chatbot icon (message bubble) in the bottom-right corner.

Click "Get Intervention Advice" to receive personalized suggestions powered by the Gemini LLM.

## Organizing Committee
## Omnitrics Hackathon 2025
## National Degree College, Basavanagudi
