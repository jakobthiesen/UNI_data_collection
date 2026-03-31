import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("rgb_data.csv", header=None)

df.columns = [
    "timestamp",
    "session_id",
    "repeat_id",
    "distance_mm",
    "target_id",
    "current_uA",
    "sample_index",
    "ambient",
    "R",
    "G",
    "B",
    "total"
]

# -----------------------------
# Force numeric types
# -----------------------------
numeric_cols = [
    "timestamp",
    "session_id",
    "repeat_id",
    "distance_mm",
    "current_uA",
    "sample_index",
    "ambient",
    "R",
    "G",
    "B",
    "total"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing critical values
df = df.dropna(subset=[
    "distance_mm", "current_uA", "ambient", "R", "G", "B"
]).copy()

# Recompute total safely
df["total"] = df["R"] + df["G"] + df["B"]

# Keep only physically valid rows
df = df[(df["total"] > 0) & (df["current_uA"] > 0) & (df["distance_mm"] > 0)].copy()

# -----------------------------
# Aggregate per repeat-condition
# -----------------------------
grouped = df.groupby(
    ["session_id", "repeat_id", "target_id", "distance_mm", "current_uA"],
    as_index=False
).mean(numeric_only=True)

# Recompute total after grouping
grouped["total"] = grouped["R"] + grouped["G"] + grouped["B"]
grouped = grouped[
    (grouped["total"] > 0) &
    (grouped["current_uA"] > 0) &
    (grouped["distance_mm"] > 0)
].copy()

# -----------------------------
# Feature engineering
# -----------------------------
eps = 1e-9

grouped["log_total"] = np.log(grouped["total"] + eps)
grouped["log_current"] = np.log(grouped["current_uA"] + eps)
grouped["log_ambient"] = np.log(grouped["ambient"] + eps)
grouped["log_distance"] = np.log(grouped["distance_mm"] + eps)

# Material / spectral information
# grouped["R_norm"] = grouped["R"] / (grouped["total"] + eps)
# grouped["G_norm"] = grouped["G"] / (grouped["total"] + eps)
# grouped["B_norm"] = grouped["B"] / (grouped["total"] + eps)
grouped["R_norm"] = np.log(grouped["R"]+ eps) / (grouped["log_total"] + eps)
grouped["G_norm"] = np.log(grouped["G"]+ eps) / (grouped["log_total"] + eps)
grouped["B_norm"] = np.log(grouped["B"]+ eps) / (grouped["log_total"] + eps)


grouped["R_over_G"] = grouped["R"] / (grouped["G"] + eps)
grouped["R_over_B"] = grouped["R"] / (grouped["B"] + eps)
grouped["G_over_B"] = grouped["G"] / (grouped["B"] + eps)

# -----------------------------
# Select features
# -----------------------------
features = [
    "log_total",
    "log_current",
    "R_norm",
    "G_norm",
    "B_norm",
    "R_over_G",
    "R_over_B",
    "G_over_B",
    "log_ambient"
]

X = grouped[features]
y = grouped["log_distance"]   # <-- LOG target

# Keep original mm target for later reporting
y_mm = grouped["distance_mm"]

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test, y_train_mm, y_test_mm = train_test_split(
    X, y, y_mm,
    test_size=0.2,
    random_state=42,
    stratify=grouped["distance_mm"]
)

# -----------------------------
# XGBoost regressor
# -----------------------------
model = xgb.XGBRegressor(
    n_estimators=100000,
    max_depth=16,
    learning_rate=0.01,
    min_child_weight=5,
    gamma = 0.05,
    reg_alpha = 0.2,
    reg_lambda = 1.5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

# Train on log-distance
model.fit(X_train, y_train)

# Predict in log-space
y_pred_log = model.predict(X_test)

# Convert back to mm
y_pred_mm = np.exp(y_pred_log)
y_test_mm = np.array(y_test_mm)

# -----------------------------
# Evaluation in mm
# -----------------------------
mae = mean_absolute_error(y_test_mm, y_pred_mm)
rmse = np.sqrt(mean_squared_error(y_test_mm, y_pred_mm))

print(f"Number of grouped samples: {len(grouped)}")
print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"MAE:  {mae:.2f} mm")
print(f"RMSE: {rmse:.2f} mm")

print("Train distances:", sorted(np.unique(y_train_mm)))
print("Test distances:", sorted(np.unique(y_test_mm)))


model.save_model("distance_model.json")

# -----------------------------
# Plot: True vs Predicted
# -----------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test_mm, y_pred_mm, alpha=0.7)
plt.plot([y_mm.min(), y_mm.max()], [y_mm.min(), y_mm.max()], "r--")
plt.xlabel("True Distance (mm)")
plt.ylabel("Predicted Distance (mm)")
plt.title("XGBoost Log-Log Model")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Plot: Absolute Error vs Distance
# -----------------------------
error = np.abs(y_test_mm - y_pred_mm)

plt.figure(figsize=(7, 6))
plt.scatter(y_test_mm, error, alpha=0.7)
plt.xlabel("Distance (mm)")
plt.ylabel("Absolute Error (mm)")
plt.title("Absolute Error vs Distance")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Plot: Residual vs Distance
# -----------------------------
residual = y_pred_mm - y_test_mm

plt.figure(figsize=(7, 6))
plt.scatter(y_test_mm, residual, alpha=0.7)
plt.axhline(0, linestyle="--")
plt.xlabel("True Distance (mm)")
plt.ylabel("Residual (Predicted - True) (mm)")
plt.title("Residual vs Distance")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Feature importance
# -----------------------------
importance = model.feature_importances_
order = np.argsort(importance)[::-1]

plt.figure(figsize=(8, 5))
plt.bar(np.array(features)[order], importance[order])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()