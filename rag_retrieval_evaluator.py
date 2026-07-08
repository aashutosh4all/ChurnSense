import time
import pandas as pd

# Import YOUR retrieval function
from app.app_standalone import retrieve_relevant_policies


# -------------------------------------------------------
# Load Evaluation Dataset
# -------------------------------------------------------

df = pd.read_csv("rag_eval_dataset.csv")


results = []


# -------------------------------------------------------
# Evaluate every customer
# -------------------------------------------------------

for _, row in df.iterrows():

    customer = {
        "CreditScore": row["CreditScore"],
        "Age": row["Age"],
        "Tenure": row["Tenure"],
        "Balance": row["Balance"],
        "NumOfProducts": row["NumOfProducts"],
        "HasCrCard": row["HasCrCard"],
        "IsActiveMember": row["IsActiveMember"],
        "EstimatedSalary": row["EstimatedSalary"],
        "Country": row["Country"]
    }

    expected = [
        p.strip()
        for p in str(row["ExpectedPolicies"]).split(";")
    ]

    # -----------------------------
    # Retrieval Latency
    # -----------------------------

    start = time.perf_counter()

    retrieved = retrieve_relevant_policies(customer)

    latency = (time.perf_counter() - start) * 1000


    retrieved_ids = []

    for p in retrieved:

        for policy_id, policy in retrieve_relevant_policies.__globals__["BANK_POLICIES"].items():

            if policy["title"] == p["title"]:
                retrieved_ids.append(policy_id)
                break

    # -------------------------------------------------------
    # Retrieval Metrics
    # -------------------------------------------------------

    k = min(3, len(retrieved_ids))

    top_k = retrieved_ids[:k]

    # Hit@1
    hit_at_1 = int(
        len(retrieved_ids) > 0 and
        retrieved_ids[0] in expected
    )

    # Hit@3
    hit_at_3 = int(
        any(policy in expected for policy in top_k)
    )

    # Precision@3
    precision_at_3 = (
        len(set(top_k).intersection(expected))
        / k
    ) if k > 0 else 0

    # Recall@3
    recall_at_3 = (
        len(set(top_k).intersection(expected))
        / len(expected)
    ) if len(expected) > 0 else 0

    # Mean Reciprocal Rank
    mrr = 0

    for rank, policy in enumerate(retrieved_ids, start=1):

        if policy in expected:
            mrr = 1 / rank
            break


    results.append({
        "customer_id": row["customer_id"],
        "scenario": row["scenario"],

        "expected": ";".join(expected),
        "retrieved": ";".join(retrieved_ids),

        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,

        "precision_at_3": round(precision_at_3,3),
        "recall_at_3": round(recall_at_3,3),
        "mrr": round(mrr,3),

        "latency_ms": round(latency,2)
    })


results = pd.DataFrame(results)

# -------------------------------------------------------
# Save Detailed Results
# -------------------------------------------------------

results.to_csv(
    "rag_retrieval_results.csv",
    index=False
)

# -------------------------------------------------------
# Overall Retrieval Metrics
# -------------------------------------------------------

summary = {
    "Hit@1": results["hit_at_1"].mean(),
    "Hit@3": results["hit_at_3"].mean(),
    "Precision@3": results["precision_at_3"].mean(),
    "Recall@3": results["recall_at_3"].mean(),
    "MRR": results["mrr"].mean(),
    "Average Latency (ms)": results["latency_ms"].mean()
}

summary_df = pd.DataFrame([summary])

summary_df.to_csv(
    "rag_retrieval_summary.csv",
    index=False
)

print("=" * 60)
print("RETRIEVAL EVALUATION SUMMARY")
print("=" * 60)
print(summary_df.round(3))

print()

print("Detailed Results Saved  -> rag_retrieval_results.csv")
print("Summary Saved           -> rag_retrieval_summary.csv")