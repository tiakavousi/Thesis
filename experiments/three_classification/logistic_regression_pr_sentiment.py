import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2

# Create report directory 
REPORT_DIR = "report"
os.makedirs(REPORT_DIR, exist_ok=True)

# Load and prepare data 
df = pd.read_csv("data/pr_sentiment_aggregated/pr_sentiment_summary.csv")
df["pr_outcome_binary"] = df["pr_outcome"].map({"merged": 1, "rejected": 0})

features = ["weighted_negativity_ratio", "has_any_negative_feedback"]
X = sm.add_constant(df[features])
y = df["pr_outcome_binary"]

# Fit logistic regression model 
model = sm.Logit(y, X)
result = model.fit()

# Save full model summary 
with open(os.path.join(REPORT_DIR, "logistic_regression_summary.txt"), "w") as f:
    f.write(result.summary().as_text())

# Calculate and save odds ratios + confidence intervals
odds_ratios = np.exp(result.params)
conf_int = np.exp(result.conf_int())
odds_df = pd.DataFrame({
    "Odds Ratio": odds_ratios,
    "CI Lower (2.5%)": conf_int[0],
    "CI Upper (97.5%)": conf_int[1]
})
odds_df.to_csv(os.path.join(
    REPORT_DIR, 
    "odds_ratios_and_confidence_intervals.csv"
    ))

# Predict probabilities for Hosmer-Lemeshow test
y_pred_proba = result.predict(X)
# Hosmer-Lemeshow Test 
def hosmer_lemeshow_test(y_true, y_prob, g=10):
    df_hl = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df_hl["decile"] = pd.qcut(df_hl["y_prob"], q=g, duplicates="drop")
    grouped = df_hl.groupby("decile")

    obs = grouped["y_true"].agg(["sum", "count"])
    obs.columns = ["observed_events", "total"]
    obs["observed_nonevents"] = obs["total"] - obs["observed_events"]
    obs["expected_events"] = grouped["y_prob"].sum()
    obs["expected_nonevents"] = obs["total"] - obs["expected_events"]

    chisq = (
        ((obs["observed_events"] - obs["expected_events"]) ** 2 / obs["expected_events"]) +
        ((obs["observed_nonevents"] - obs["expected_nonevents"]) ** 2 / obs["expected_nonevents"])
    ).sum()

    df_ = len(obs) - 2
    p_value = 1 - chi2.cdf(chisq, df_)

    return chisq, p_value

hl_stat, hl_pval = hosmer_lemeshow_test(y, y_pred_proba)

with open(os.path.join(REPORT_DIR, "hosmer_lemeshow_test.txt"), "w") as f:
    f.write(f"Hosmer-Lemeshow Statistic: {hl_stat:.4f}\n")
    f.write(f"p-value: {hl_pval:.4f}\n")
    f.write("Interpretation: " + (
        "GOOD fit (fail to reject H0)\n" if hl_pval > 0.05 else "POOR fit (reject H0)\n"
    ))
