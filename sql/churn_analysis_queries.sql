CREATE DATABASE churn_analysis;
USE churn_analysis;
SELECT * 
FROM bank_churn
LIMIT 5;

SELECT 
    Exited,
    COUNT(*) AS total_customers,
    ROUND(COUNT(*) * 100.0 / (SELECT 
                    COUNT(*)
                FROM
                    bank_churn),
            2) AS churn_percentage
FROM
    bank_churn
GROUP BY Exited;

SELECT 
    Geography,
    COUNT(*) AS total_customers,
    SUM(Exited) AS churned_customers,
    ROUND(SUM(Exited) * 100.0 / COUNT(*), 2) AS churn_rate
FROM bank_churn
GROUP BY Geography
ORDER BY churn_rate DESC;

SELECT 
    IsActiveMember,
    COUNT(*) AS total_customers,
    SUM(Exited) AS churned_customers,
    ROUND(SUM(Exited) * 100.0 / COUNT(*), 2) AS churn_rate
FROM bank_churn
GROUP BY IsActiveMember;

SELECT 
    Geography,
    AgeGroup,
    IsActiveMember,
    COUNT(*) AS total_customers,
    SUM(Exited) AS churned_customers,
    ROUND(SUM(Exited) * 100.0 / COUNT(*), 2) AS churn_rate
FROM bank_churn
WHERE Geography = 'Germany'
GROUP BY Geography, AgeGroup, IsActiveMember
ORDER BY churn_rate DESC;

SELECT *
FROM bank_churn
WHERE AgeGroup='';

SET SQL_SAFE_UPDATES=0;
UPDATE bank_churn
SET AgeGroup = '18-30'
WHERE Age = 18;
SET SQL_SAFE_UPDATES=1;
SELECT COUNT(*) AS empty_age_groups
FROM bank_churn
WHERE AgeGroup = '' OR AgeGroup IS NULL;

/*TOP 10 HIGH RISK CUSTOMER SEGMENT ANALYSIS*/
WITH segment_churn AS (
    SELECT
        Geography,
        AgeGroup,
        IsActiveMember,
        NumOfProducts,
        COUNT(*) AS total_customers,
        SUM(Exited) AS churned_customers,
        ROUND(SUM(Exited) * 100.0 / COUNT(*), 2) AS churn_rate
    FROM bank_churn
    GROUP BY Geography, AgeGroup, IsActiveMember, NumOfProducts
)

SELECT *
FROM segment_churn
WHERE total_customers >= 30
ORDER BY churn_rate DESC
LIMIT 10;

/*REVENUE AT RISK ANALYSIS*/
WITH churn_risk AS (
    SELECT
        Geography,
        COUNT(*) AS total_customers,
        SUM(Exited) AS churned_customers,
        ROUND(SUM(Balance), 2) AS total_balance,
        ROUND(SUM(CASE WHEN Exited = 1 THEN Balance ELSE 0 END), 2) AS churned_balance
    FROM bank_churn
    GROUP BY Geography
)

SELECT
    Geography,
    total_customers,
    churned_customers,
    ROUND(churned_customers * 100.0 / total_customers, 2) AS churn_rate,
    total_balance,
    churned_balance,
    ROUND(churned_balance * 100.0 / total_balance, 2) AS balance_at_risk_pct
FROM churn_risk
ORDER BY balance_at_risk_pct DESC;

/*AGE WISE SEGMENTATION OF ALL COUNTRIES: AGE GROUPS WITH TOP-2 CHURN RATE OF ALL COUNTRIES*/
WITH age_geo_churn AS (
    SELECT
        Geography,
        AgeGroup,
        COUNT(*) AS total_customers,
        SUM(Exited) AS churned_customers,
        ROUND(SUM(Exited) * 100.0 / COUNT(*), 2) AS churn_rate
    FROM bank_churn
    GROUP BY Geography, AgeGroup
),

ranked_segments AS (
    SELECT
        *,
        RANK() OVER (
            PARTITION BY Geography
            ORDER BY churn_rate DESC
        ) AS churn_rank
    FROM age_geo_churn
)

SELECT *
FROM ranked_segments
WHERE churn_rank <= 2
ORDER BY Geography, churn_rank;

/*RISK SCORE ANALYSIS*/
WITH customer_scores AS (
    SELECT *,
        CASE WHEN Geography = 'Germany' THEN 2 ELSE 0 END +
        CASE WHEN Age >= 40 THEN 2 ELSE 0 END +
        CASE WHEN IsActiveMember = 0 THEN 2 ELSE 0 END +
        CASE WHEN Balance > 90000 THEN 1 ELSE 0 END +
        CASE WHEN NumOfProducts > 2 THEN 3 ELSE 0 END
        AS risk_score
    FROM bank_churn
)

SELECT
    risk_score,
    COUNT(*) AS customers,
    SUM(Exited) AS churned,
    ROUND(SUM(Exited) * 100.0 / COUNT(*), 2) AS churn_rate
FROM customer_scores
GROUP BY risk_score
ORDER BY risk_score DESC;

/*TOP-20 HIGH VALUE CHURNED CUSTOMERS*/
SELECT *
FROM bank_churn
WHERE Exited = 1
  AND Balance > (SELECT AVG(Balance) FROM bank_churn)
ORDER BY Balance DESC
LIMIT 20;

/*RETENTION TARGET ANALYSIS*/
WITH risk_segment AS (
    SELECT *
    FROM bank_churn
    WHERE Geography = 'Germany'
      AND IsActiveMember = 0
      AND Age >= 40
      AND Balance > 90000
)

SELECT
    COUNT(*) AS target_customers,
    SUM(Exited) AS churned_customers,
    ROUND(SUM(Exited) * 100.0 / COUNT(*), 2) AS churn_rate,
    ROUND(AVG(Balance), 2) AS avg_balance
FROM risk_segment;

WITH churn_contribution AS (
    SELECT
        Geography,
        COUNT(*) AS total_customers,
        SUM(Exited) AS churned_customers
    FROM bank_churn
    GROUP BY Geography
)

SELECT
    Geography,
    churned_customers,
    ROUND(
        churned_customers * 100.0 /
        (SELECT SUM(Exited) FROM bank_churn),
        2
    ) AS contribution_pct
FROM churn_contribution
ORDER BY contribution_pct DESC;

SELECT
    NumOfProducts,
    COUNT(*) AS total_customers,
    SUM(Exited) AS churned_customers,
    ROUND(SUM(Exited) * 100.0 / COUNT(*),2) AS churn_rate
FROM bank_churn
GROUP BY NumOfProducts
ORDER BY NumOfProducts;

WITH balance_deciles AS (
    SELECT
        *,
        NTILE(10) OVER (ORDER BY Balance) AS balance_decile
    FROM bank_churn
)

SELECT
    balance_decile,
    COUNT(*) AS customers,
    ROUND(MIN(Balance), 2) AS min_balance,
    ROUND(MAX(Balance), 2) AS max_balance,
    SUM(Exited) AS churned,
    ROUND(SUM(Exited) * 100.0 / COUNT(*), 2) AS churn_rate
FROM balance_deciles
GROUP BY balance_decile
ORDER BY balance_decile;

/*AGE GROUP AND BALANCE CATEGORY GROUP ANALYSIS*/
WITH customer_bands AS (
    SELECT
        *,
        CASE
            WHEN Balance = 0 THEN 'Zero Balance'
            WHEN Balance < 50000 THEN 'Low Balance'
            WHEN Balance < 100000 THEN 'Medium Balance'
            ELSE 'High Balance'
        END AS balance_band
    FROM bank_churn
)

SELECT
    AgeGroup,
    balance_band,
    COUNT(*) AS customers,
    SUM(Exited) AS churned,
    ROUND(SUM(Exited) * 100.0 / COUNT(*), 2) AS churn_rate
FROM customer_bands
GROUP BY AgeGroup, balance_band
HAVING COUNT(*) >= 50
ORDER BY churn_rate DESC;

WITH customer_bands AS (
    SELECT
        *,
        CASE
            WHEN Balance = 0 THEN 'Zero Balance'
            WHEN Balance < 50000 THEN 'Low Balance'
            WHEN Balance < 100000 THEN 'Medium Balance'
            ELSE 'High Balance'
        END AS balance_band
    FROM bank_churn
)
SELECT
    NumOfProducts,
    balance_band,
    COUNT(*) AS total_customers,
    SUM(Exited) AS churned_customers,
    ROUND(SUM(Exited) * 100.0 / COUNT(*), 2) AS churn_rate
FROM customer_bands
GROUP BY NumOfProducts, balance_band
HAVING total_customers >= 30
ORDER BY churn_rate DESC;

WITH segment_churn AS (
    SELECT
        Geography,
        AgeGroup,
        COUNT(*) AS total_customers,
        SUM(Exited) AS churned_customers,
        ROUND(SUM(Exited) * 100.0 / COUNT(*), 2) AS churn_rate
    FROM bank_churn
    GROUP BY Geography, AgeGroup
)

SELECT
    Geography,
    AgeGroup,
    total_customers,
    churned_customers,
    churn_rate,
    ROUND(
        churned_customers * 100.0 / 
        (SELECT SUM(Exited) FROM bank_churn),
        2
    ) AS churn_contribution_pct
FROM segment_churn
WHERE total_customers >= 50
ORDER BY churn_contribution_pct DESC;