import pandas as pd

# === CONFIG ===
INPUT_FILE = "notebooks/fine_tuning/fine_tuning_3class/saved_models/evaluation_results/test_predictions_with_confidence.csv"
LOW_CONF_THRESHOLD = 0.60
HIGH_CONF_THRESHOLD = 0.90

# === Load the data ===
df = pd.read_csv(INPUT_FILE)

# === Add column to mark correct predictions ===
df["correct"] = df["true_index"] == df["predicted_index"]

# === Define confidence bands ===
def label_conf_band(conf):
    if conf < LOW_CONF_THRESHOLD:
        return "Low (<0.60)"
    elif conf < HIGH_CONF_THRESHOLD:
        return "Medium (0.60–0.90)"
    else:
        return "High (>=0.90)"

df["confidence_band"] = df["confidence"].apply(label_conf_band)

# === Group by confidence band ===
grouped = df.groupby("confidence_band")["correct"].agg([
    ("total", "count"),
    ("correct", "sum"),
    ("incorrect", lambda x: (~x).sum()),
    ("accuracy", lambda x: x.mean() * 100)
])

# === Print analysis ===
print("\n📊 Confidence Band Analysis:")
print(grouped)

# === Optional: Show some misclassified examples by confidence band ===
print("\n🔍 Example Low-Confidence Misclassifications:")
print(df[(df["confidence_band"] == "Low (<0.60)") & (~df["correct"])][
    ["message", "true_label", "predicted_label", "confidence"]
].head(5))

print("\n🔍 Example High-Confidence Misclassifications:")
print(df[(df["confidence_band"] == "High (>=0.90)") & (~df["correct"])][
    ["message", "true_label", "predicted_label", "confidence"]
].sort_values("confidence", ascending=False).head(5))
