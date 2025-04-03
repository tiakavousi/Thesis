import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from pathlib import Path

# === Step 3 (Enhanced): Logistic Regression with Balancing and Multiple Predictors ===

# Paths
INPUT_FILE = Path("/Users/tayebekavousi/Desktop/github_sa/data/features/all_repo_features.csv")
OUTPUT_FILE = Path("/Users/tayebekavousi/Desktop/github_sa/data/features/model_results_enhanced.txt")

# Load data
df = pd.read_csv(INPUT_FILE)

# Compute sentiment ratios before using them
df["comment_neg_ratio"] = df["neg_comments"] / df["total_comments"]
df["review_neg_ratio"] = df["neg_reviews"] / df["total_reviews"]
df["review_comment_neg_ratio"] = df["neg_review_comments"] / df["total_review_comments"]

# Drop rows with missing values due to division by zero
required_cols = ["outcome", "comment_neg_ratio", "review_neg_ratio", "review_comment_neg_ratio"]
df = df.dropna(subset=required_cols)


# Balance the dataset by undersampling the majority class (outcome=1)
merged_df = df[df["outcome"] == 1]
closed_df = df[df["outcome"] == 0]

# Random undersample merged PRs to match the number of closed PRs
merged_sampled = merged_df.sample(n=len(closed_df), random_state=42)
df_balanced = pd.concat([merged_sampled, closed_df], ignore_index=True)

# Standardize the predictors
features = ["comment_neg_ratio", "review_neg_ratio", "review_comment_neg_ratio"]
scaler = StandardScaler()
df_balanced[[f + "_std" for f in features]] = scaler.fit_transform(df_balanced[features])

# Prepare model inputs
X = sm.add_constant(df_balanced[[f + "_std" for f in features]])  # Add intercept
y = df_balanced["outcome"]

# Fit logistic regression model
model = sm.Logit(y, X).fit()

# Predict
y_pred_prob = model.predict(X)
y_pred_class = (y_pred_prob >= 0.5).astype(int)

# Evaluation metrics
conf_matrix = confusion_matrix(y, y_pred_class)
accuracy = accuracy_score(y, y_pred_class)

# Extract odds ratios and p-values
odds_ratios = np.exp(model.params)
p_values = model.pvalues

# Save results
with open(OUTPUT_FILE, "w") as f:
    f.write("=== Logistic Regression Summary (Balanced Dataset) ===\n")
    f.write(str(model.summary()))
    f.write("\n\n=== Evaluation Metrics ===\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Confusion Matrix:\n{conf_matrix}\n")
    f.write("\n=== Odds Ratios ===\n")
    for param in model.params.index:
        f.write(f"{param}: {odds_ratios[param]:.4f} (p={p_values[param]:.4e})\n")

print(f"✅ Enhanced model results saved to: {OUTPUT_FILE}")
print(f"📊 Accuracy: {accuracy:.4f}")
print("📈 Odds Ratios:")
for param in model.params.index:
    print(f" - {param}: {odds_ratios[param]:.4f} (p={p_values[param]:.4e})")
