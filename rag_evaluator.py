import os
import time
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

# ── 1. Setup & Auth ──
load_dotenv()
API_KEY = os.environ.get("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("API Key not found. Please check your .env file.")

client = Groq(api_key=API_KEY)

# ── 2. The Bank Policies & Function (Copied here to avoid Streamlit import errors) ──
BANK_POLICIES = """
- High-Balance Customers (Balance > $100k): Eligible for 'Premium Wealth' tier, which includes a dedicated financial advisor and 0% wire fees.
- Loyal Customers (Tenure >= 5 years): Eligible for a lifetime fee-free credit card and a 0.5% APY bonus on all savings accounts.
- Inactive Customers (Active = No): Eligible for a 'Reactivation Campaign' offering a $200 cash bonus for setting up direct deposit.
- Senior Customers (Age >= 55): Eligible for 'Senior Priority Banking', offering free cashier's checks, estate planning consultations, and higher CD yields.
- Multi-Product Customers (Products >= 3): Eligible for consolidated relationship pricing (discounts on mortgage and auto loan rates).
"""

def generate_retention_strategy(customer_data, top_drivers):
    drivers_text = "\n".join([f"- {f} (Impact: +{imp:.3f})" for f, imp in top_drivers])
    
    prompt = f"""
    You are a Senior Bank Customer Retention Expert. A high-risk customer has been flagged for potential churn.
    
    Customer Profile:
    - Age: {customer_data['Age']}
    - Balance: ${customer_data['Balance']:,.2f}
    - Tenure: {customer_data['Tenure']} years
    - Products: {customer_data['NumOfProducts']}
    - Active Member: {'Yes' if customer_data['IsActiveMember'] else 'No'}
    
    Top reasons the AI flagged this customer for churn (SHAP values):
    {drivers_text}
    
    Bank Internal Policies & Offers:
    {BANK_POLICIES}
    
    Task: 
    1. Analyze the Customer Profile and the SHAP churn drivers.
    2. Select the most relevant offer from the "Bank Internal Policies" that specifically addresses their churn risk.
    3. Write a highly personalized, 2-to-3 sentence strategy for the account manager to use. 
    
    Do not use pleasantries or greetings. Be specific, actionable, and explicitly mention the bank policy you are applying.
    """
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=200
    )
    return response.choices[0].message.content

# ── 3. The Golden Test Cases ──
test_cases = [
    {
        "name": "High Balance & Senior",
        "data": {"Age": 62, "Balance": 150000, "Tenure": 2, "NumOfProducts": 1, "IsActiveMember": 1},
        "drivers": [("Balance", 0.15), ("Age", 0.10)]
    },
    {
        "name": "Loyal but Inactive",
        "data": {"Age": 35, "Balance": 20000, "Tenure": 6, "NumOfProducts": 1, "IsActiveMember": 0},
        "drivers": [("IsActiveMember", 0.20), ("Tenure", -0.05)]
    },
    {
        "name": "Multi-Product Risk",
        "data": {"Age": 40, "Balance": 50000, "Tenure": 3, "NumOfProducts": 4, "IsActiveMember": 1},
        "drivers": [("NumOfProducts", 0.25)]
    }
]

# ── 4. The LLM Judge ──
def evaluate_rag_response(customer, strategy, policies):
    judge_prompt = f"""
    You are an impartial AI evaluator grading a Bank Retention System.
    
    Customer Data: {customer}
    Bank Policies (Context): {policies}
    Generated Strategy (Answer): {strategy}
    
    Grade the Generated Strategy strictly from 1 to 5 on these three metrics:
    1. Faithfulness: Does the strategy ONLY use offers explicitly mentioned in the Bank Policies without hallucinating? (1=Hallucinated, 5=Strictly factual).
    2. Relevance: Does the strategy directly address the customer's specific profile and risk factors? (1=Generic, 5=Highly personalized).
    3. Policy Match: Did it select the CORRECT policy for this user? (1=Wrong policy, 5=Perfect match).
    
    Output your response ONLY in this exact comma-separated format: [Faithfulness Score],[Relevance Score],[Policy Match Score]
    Example: 5,4,5
    """
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": judge_prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        max_tokens=10
    )
    return response.choices[0].message.content.strip()

# ── 5. Run the Evaluation Pipeline ──
print("🚀 Starting RAG Evaluation Pipeline...\n")
results = []

for idx, tc in enumerate(test_cases):
    print(f"Testing Case {idx+1}: {tc['name']}...")
    
    start_time = time.time()
    strategy = generate_retention_strategy(tc['data'], tc['drivers'])
    latency = time.time() - start_time
    
    scores = evaluate_rag_response(tc['data'], strategy, BANK_POLICIES)
    
    try:
        f_score, r_score, p_score = map(int, scores.split(','))
    except:
        f_score, r_score, p_score = "Error", "Error", "Error"
        
    results.append({
        "Test Case": tc['name'],
        "Latency (sec)": round(latency, 2),
        "Faithfulness (1-5)": f_score,
        "Relevance (1-5)": r_score,
        "Policy Match (1-5)": p_score,
    })

# ── 6. Export Results ──
df_results = pd.DataFrame(results)
print("\n✅ Evaluation Complete! Here are the metrics:\n")
print(df_results.to_markdown(index=False))
df_results.to_csv("rag_evaluation_metrics.csv", index=False)