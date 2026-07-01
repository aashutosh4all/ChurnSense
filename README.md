# ChurnSense AI: Predictive & Generative Customer Retention

End-to-end customer churn prediction and retention analytics project using Python, SQL, Machine Learning, and Generative AI (RAG).

## Project Overview

ChurnSense AI is an intelligent retention platform that moves beyond simple churn prediction. It combines predictive machine learning with Retrieval-Augmented Generation (RAG) to provide actionable, policy-driven retention strategies for at-risk customers. The system identifies key churn drivers, visualizes business insights, and generates specific manager interventions based on internal bank policies.

## System Architecture

### 1. Predictive Engine (Machine Learning)
- **Model:** Random Forest Classifier (optimized at a 0.4 decision threshold).
- **Explainability:** SHAP (SHapley Additive exPlanations) values identify the exact mathematical features (e.g., Age, Tenure, Product Usage) driving an individual's churn risk.
- **Analytics:** SQL-driven exploratory data analysis and Power BI segmentation.

### 2. Generative Retention Engine (RAG)
- **Framework:** Retrieval-Augmented Generation (RAG).
- **LLM:** Groq API (Llama-3.3-70b-versatile).
- **Intelligence:** The system maps the specific SHAP-identified risk factors against an internal Knowledge Base of Bank Policies. It then generates personalized, hallucination-free retention strategies for account managers to execute.

## Objectives

- Analyze customer churn patterns using Python and SQL.
- Identify major churn drivers through exploratory data analysis.
- Build and evaluate machine learning models for churn prediction.
- Implement Explainable AI (SHAP) to interpret individual predictions.
- Develop a RAG pipeline to generate context-aware retention strategies.
- Create an interactive Power BI dashboard for business insights.
- Deploy a Streamlit web app for end-to-end prediction and generation.

## Tools and Technologies

- **Languages:** Python, SQL
- **Machine Learning:** Scikit-learn, Pandas, NumPy, SHAP
- **Generative AI:** Groq API, Llama-3, python-dotenv
- **Database & Viz:** MySQL, Power BI, Matplotlib, Seaborn, Plotly
- **Deployment & API:** FastAPI, Streamlit

## RAG System Evaluation Metrics

To ensure the reliability of the AI-generated retention strategies, the RAG pipeline was evaluated using an LLM-as-a-Judge framework on edge-case customer profiles:

| Test Case             | Latency (sec) | Faithfulness (1-5) | Relevance (1-5) | Policy Match (1-5) |
|:----------------------|--------------:|-------------------:|----------------:|-------------------:|
| High Balance & Senior | 6.43          | 5                  | 5               | 5                  |
| Loyal but Inactive    | 0.76          | 5                  | 5               | 5                  |
| Multi-Product Risk    | 0.76          | 5                  | 4               | 5                  |

*Note: Initial latency reflects connection overhead; subsequent generations operate at sub-second speeds. High Faithfulness scores indicate strict adherence to internal policies without hallucination.*

## Key Insights

- **Overall churn rate** was approximately 20.37%.
- **Germany** showed the highest churn rate among all regions.
- **Inactive customers** churned at a much higher rate than active customers.
- Customers **aged 51–60** showed the highest churn tendency.
- Customers with **more than two products** showed unusually high churn rates.
- **GenAI Application:** The RAG system successfully maps these specific demographic and product risks directly to actionable policies (e.g., matching Multi-Product risk to Consolidated Relationship Pricing).

## Power BI Dashboard

### Churn Overview
![Churn Overview](images/dashboard_p1_overview.png)

### Customer Segmentation Analysis
![Customer Segmentation](images/dashboard_p2_segmentation.png)

### Risk and Retention Analysis
![Risk Analysis](images/dashboard_p3_risk_analysis.png)

## Streamlit Web App

The Streamlit app acts as the front-end for the dual-engine system, providing:
- Churn probability and risk classification.
- SHAP feature importance visualization.
- AI-generated, policy-backed retention recommendations.

![Streamlit App](images/streamlit_app.png)

## How to Run the Application

### 1. Streamlit Web App
Navigate to the app folder:
```bash
cd app
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Set up your environment variables by creating a `.env` file in the root directory:
```text
GROQ_API_KEY=your_api_key_here
```

Run the standalone Streamlit app:
```bash
python -m streamlit run app_standalone.py
```

### 2. FastAPI Backend
To run the FastAPI backend, ensure you are in the `app` directory:
```bash
python -m uvicorn main:app --reload
```
API documentation will be available at: http://127.0.0.1:8000/docs

## Project Structure

```text
customer-churn-retention-analysis/
│
├── app/
│   ├── artifacts/             # Trained ML models and feature mapping
│   ├── app_standalone.py      # Main Streamlit App + GenAI RAG Logic
│   ├── app.py
│   ├── main.py                # FastAPI Backend
│   ├── requirements.txt
│   └── RUN_APP.md
│
├── dashboard/
│   └── bank_churn_dashboard.pbix
│
├── data/
│   └── bank_churn_cleaned.csv
│
├── images/
│   ├── dashboard_p1_overview.png
│   ├── dashboard_p2_segmentation.png
│   ├── dashboard_p3_risk_analysis.png
│   └── streamlit_app.png
│
├── notebooks/
│   ├── churn_eda_ml_analysis.ipynb
│   └── final_model_training.ipynb
│
├── sql/
│   └── churn_analysis_queries.sql
│
├── rag_evaluator.py           # GenAI automated evaluation script
├── requirements.txt
├── README.md
└── LICENSE
```

### Conclusion
This project integrates traditional machine learning with modern Generative AI to provide a comprehensive retention solution. - - By combining SHAP-based explainability with LLM-generated strategies, ChurnSense AI enables data-driven decisions that are both accurate and strictly aligned with business policy.