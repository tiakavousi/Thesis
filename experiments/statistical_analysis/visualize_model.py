import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# === Step 4: Visualizations and Evaluation ===

# Paths
INPUT_FILE = Path("/Users/tayebekavousi/Desktop/github_sa/data/features/all_repo_features.csv")
FIGURES_DIR = Path("/Users/tayebekavousi/Desktop/github_sa/data/features/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(INPUT_FILE)

# Compute missing ratio columns
df["comment_neg_ratio"] = df["neg_comments"] / df["total_comments"]
df["review_neg_ratio"] = df["neg_reviews"] / df["total_reviews"]
df["review_comment_neg_ratio"] = df["neg_review_comments"] / df["total_review_comments"]
df = df.dropna(subset=["outcome", "comment_neg_ratio", "review_neg_ratio", "review_comment_neg_ratio"])

# Balance dataset
merged_df = df[df["outcome"] == 1]
closed_df = df[df["outcome"] == 0]
merged_sampled = merged_df.sample(n=len(closed_df), random_state=42)
df_balanced = pd.concat([merged_sampled, closed_df], ignore_index=True)

# Standardize features
features = ["comment_neg_ratio", "review_neg_ratio", "review_comment_neg_ratio"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_balanced[features])
X = pd.DataFrame(X_scaled, columns=[f + "_std" for f in features])
X["const"] = 1.0
y = df_balanced["outcome"]

# Fit model
model = sm.Logit(y, X).fit(disp=False)
y_pred_prob = model.predict(X)
y_pred_class = (y_pred_prob >= 0.5).astype(int)

# === ROC Curve ===
fpr, tpr, _ = roc_curve(y, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "roc_curve.png")
plt.close()

# === Confusion Matrix Heatmap ===
conf_matrix = confusion_matrix(y, y_pred_class)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Closed", "Merged"], yticklabels=["Closed", "Merged"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "confusion_matrix.png")
plt.close()

# === Coefficient Plot ===
coef_df = pd.DataFrame({
    "feature": model.params.index,
    "coef": model.params.values,
    "ci_lower": model.conf_int()[0],
    "ci_upper": model.conf_int()[1]
})
coef_df = coef_df[coef_df["feature"] != "const"]

plt.figure(figsize=(7, 4))
sns.pointplot(data=coef_df, x="coef", y="feature", join=False)
for _, row in coef_df.iterrows():
    plt.plot([row["ci_lower"], row["ci_upper"]], [row["feature"], row["feature"]], color='gray', lw=2)
plt.axvline(0, linestyle="--", color="red")
plt.title("Logistic Regression Coefficients (w/ 95% CI)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "coefficient_plot.png")
plt.close()

# === Prediction Probability Distribution ===
plt.figure(figsize=(6, 4))
sns.histplot(y_pred_prob[y == 1], color="green", label="Merged", kde=True, stat="density", bins=20)
sns.histplot(y_pred_prob[y == 0], color="red", label="Closed", kde=True, stat="density", bins=20)
plt.xlabel("Predicted Probability of Merge")
plt.title("Prediction Probability Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "probability_distribution.png")
plt.close()

print(f"✅ Visualizations saved to {FIGURES_DIR}")
