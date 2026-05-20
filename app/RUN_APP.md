# Customer Churn Prediction Web App

This folder contains the FastAPI backend and Streamlit frontend for customer churn prediction.

## 1. Install Dependencies

```bash
C:\Python313\python.exe -m pip install -r requirements.txt

## 2. Start FastAPI Backend 

Open terminal inside this folder and run:
C:\Python313\python.exe -m uvicorn main:app --reload

Backend will run at:
http://127.0.0.1:8000
API documentation:
http://127.0.0.1:8000/docs

## 3. Start Streamlit Frontend
Open a second terminal inside this folder and run:
C:\Python313\python.exe -m streamlit run app.py

Frontend will run at:
http://localhost:8501

## 4. App Workflow
Enter customer details in the Streamlit form.
The frontend sends the input to the FastAPI backend.
The backend loads the trained Random Forest model.
The model returns churn probability, prediction, risk level, and retention suggestion.