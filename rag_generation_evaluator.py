import time
import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

# Load variables from .env file into the environment
load_dotenv()

from app.app_standalone import (
    generate_retention_strategy,
    retrieve_relevant_policies,
    preprocess_input,
    model,
    explainer,
    model_columns
)

# 1. Load the dataset (removed the redundant second loading)
evaluation_df = pd.read_csv("rag_eval_dataset.csv")

print(f"Columns found: {evaluation_df.columns.tolist()}")
print(f"Starting evaluation for {len(evaluation_df)} customers...\n")

results = []

# 2. Fetch API key from environment so it doesn't fail silently
api_key = os.environ.get("GROQ_API_KEY", "")
if not api_key:
    print("⚠️ WARNING: GROQ_API_KEY environment variable not found.")
    print("The AI will return connection errors instead of real strategies.\n")

for index, row in evaluation_df.iterrows():
    # Added a print statement so you know the script isn't frozen!
    print(f"Processing Customer {row['customer_id']} ({index+1}/{len(evaluation_df)})...")
    
    customer_data = {
        "CreditScore": row["CreditScore"],
        "Country": row["Country"],
        "Gender": row["Gender"],
        "Age": row["Age"],
        "Tenure": row["Tenure"],
        "Balance": row["Balance"],
        "NumOfProducts": row["NumOfProducts"],
        "HasCrCard": row["HasCrCard"],
        "IsActiveMember": row["IsActiveMember"],
        "EstimatedSalary": row["EstimatedSalary"],
    }

    # -----------------------------
    # Predict churn probability
    # -----------------------------
    processed = preprocess_input(customer_data)

    churn_probability = float(
        model.predict_proba(processed)[0][1]
    )
    
    shap_vals = explainer.shap_values(processed)

    if isinstance(shap_vals, list):
        churn_shap_values = shap_vals[1].flatten()
    else:
        churn_shap_values = shap_vals.flatten()

    churn_shap_values = churn_shap_values[-len(model_columns):]

    feature_df = pd.DataFrame({
        "Feature": model_columns,
        "SHAP": churn_shap_values
    })

    feature_df["Value"] = [
        processed[col].iloc[0]
        for col in feature_df["Feature"]
    ]

    feature_df = feature_df[feature_df["SHAP"] > 0]
    feature_df = feature_df.sort_values(by="SHAP", ascending=False)

    top_drivers = feature_df.head(3)
    
    start = time.perf_counter()

    # -----------------------------
    # Generate Strategy
    # -----------------------------
    strategy = generate_retention_strategy(
        api_key=api_key,
        customer_data=customer_data,
        top_drivers=top_drivers,
        churn_probability=churn_probability
    )

    latency = (time.perf_counter() - start) * 1000
    
    retrieved = retrieve_relevant_policies(customer_data)
    retrieved_titles = [p["title"] for p in retrieved]
    
    results.append({
        "CustomerID": row["customer_id"], # Fixed KeyError here
        "ChurnProbability": round(churn_probability, 4),
        "RetrievedPolicies": " | ".join(retrieved_titles),
        "GeneratedStrategy": strategy,
        "GenerationLatency(ms)": round(latency, 2)
    })

results_df = pd.DataFrame(results)
results_df.to_csv("rag_generation_results.csv", index=False)

print("\n" + "=" * 60)
print("GENERATION COMPLETE")
print("=" * 60)
print(f"Customers Evaluated : {len(results_df)}")
print(f"Average Latency     : {results_df['GenerationLatency(ms)'].mean():.2f} ms")
print("Results saved -> rag_generation_results.csv")