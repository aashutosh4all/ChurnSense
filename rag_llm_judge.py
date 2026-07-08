import json
import time
import pandas as pd
from groq import Groq
import os
from dotenv import load_dotenv

# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()
API_KEY = os.environ.get("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("API Key not found. Please check your .env file.")

client = Groq(api_key=API_KEY)

# --------------------------------------------------
# Load generated responses
# --------------------------------------------------

generation_df = pd.read_csv("rag_generation_results.csv")

results = []

print(f"Starting LLM Judge evaluation for {len(generation_df)} customers...\n")

# --------------------------------------------------
# Judge every generated strategy
# --------------------------------------------------

for index, row in generation_df.iterrows():
    # BUG FIX 1: Add progress tracker
    print(f"Evaluating Customer {row['CustomerID']} ({index+1}/{len(generation_df)})...")

    prompt = f"""
You are an expert evaluator for Banking RAG systems.

Evaluate ONLY the generated retention strategy.

Customer Churn Probability:
{row["ChurnProbability"]}

Retrieved Policies:
{row["RetrievedPolicies"]}

Generated Strategy:
{row["GeneratedStrategy"]}

Score the strategy from 1 to 5 on:

1. Relevance
2. Groundedness
3. Faithfulness
4. Personalization
5. Helpfulness

Definitions:

Relevance: Does the recommendation address the customer's churn situation?
Groundedness: Is the recommendation supported by the retrieved policies?
Faithfulness: Does the strategy avoid inventing bank offers or unsupported claims?
Personalization: Does the recommendation use customer-specific information?
Helpfulness: Would a relationship manager actually find this recommendation useful?

Return ONLY valid JSON.

Example:

{{
    "relevance":5,
    "groundedness":5,
    "faithfulness":5,
    "personalization":4,
    "helpfulness":5
}}
"""

    start = time.perf_counter()

    try:

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=120,
            # BUG FIX 2: Force Groq to return guaranteed valid JSON
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        latency = (time.perf_counter() - start) * 1000

        raw = response.choices[0].message.content.strip()
        scores = json.loads(raw)

        results.append({
            "CustomerID": row["CustomerID"],
            # Using .get() ensures it doesn't crash if the LLM capitalizes a key
            "Relevance": scores.get("relevance", 0),
            "Groundedness": scores.get("groundedness", 0),
            "Faithfulness": scores.get("faithfulness", 0),
            "Personalization": scores.get("personalization", 0),
            "Helpfulness": scores.get("helpfulness", 0),
            "JudgeLatency(ms)": round(latency,2)
        })

        # BUG FIX 3: Add a short delay to prevent hitting Groq's 429 Rate Limits
        time.sleep(1)

    except Exception as e:
        print(f"Failed Customer {row['CustomerID']}: {e}")

# --------------------------------------------------
# Save results
# --------------------------------------------------

judge_df = pd.DataFrame(results)

judge_df.to_csv(
    "rag_llm_scores.csv",
    index=False
)

print("\n" + "="*60)
print("LLM JUDGE COMPLETE")
print("="*60)

print(f"Customers Evaluated : {len(judge_df)}")

print(
    f"Average Judge Latency : "
    f"{judge_df['JudgeLatency(ms)'].mean():.2f} ms"
)

print("\nSaved -> rag_llm_scores.csv")