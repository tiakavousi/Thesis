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
