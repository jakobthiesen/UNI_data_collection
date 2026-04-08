import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

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

# Keep target_id as string
df["target_id"] = df["target_id"].astype(str)

# Drop rows with missing critical values
df = df.dropna(subset=[
    "distance_mm", "current_uA", "ambient", "R", "G", "B", "target_id"
]).copy()

# # 👉 EXCLUDE BLACK HERE
# df = df[df["target_id"] != "black"].copy()

# Recompute total safely
df["total"] = df["R"] + df["G"] + df["B"]

# Keep only physically valid rows
df = df[
    (df["total"] > 0) &
    (df["current_uA"] > 0) &
    (df["distance_mm"] > 0)
].copy()

# -----------------------------
# Aggregate per repeat-condition
# -----------------------------
grouped = df.groupby(
    ["session_id", "repeat_id", "target_id", "distance_mm", "current_uA"],
    as_index=False
).mean(numeric_only=True)

grouped["total"] = grouped["R"] + grouped["G"] + grouped["B"]
grouped = grouped[
    (grouped["total"] > 0) &
    (grouped["current_uA"] > 0) &
    (grouped["distance_mm"] > 0)
].copy()

# Keep only distances 20 mm to 200 mm
grouped = grouped[
    (grouped["distance_mm"] >= 20) &
    (grouped["distance_mm"] <= 140)
].copy()

# -----------------------------
# Feature engineering
# -----------------------------
eps = 1e-9

grouped["log_total"] = np.log(grouped["total"] + eps)
grouped["log_current"] = np.log(grouped["current_uA"] + eps)
grouped["log_ambient"] = np.log(grouped["ambient"] + eps)

# Spectral features
grouped["R_norm"] = grouped["R"] / (grouped["total"] + eps)
grouped["G_norm"] = grouped["G"] / (grouped["total"] + eps)
grouped["B_norm"] = grouped["B"] / (grouped["total"] + eps)

grouped["R_over_G"] = grouped["R"] / (grouped["G"] + eps)
grouped["R_over_B"] = grouped["R"] / (grouped["B"] + eps)
grouped["G_over_B"] = grouped["G"] / (grouped["B"] + eps)

grouped["log_R_over_G"] = np.log(grouped["R_over_G"] + eps)
grouped["log_R_over_B"] = np.log(grouped["R_over_B"] + eps)
grouped["log_G_over_B"] = np.log(grouped["G_over_B"] + eps)

grouped["inv_sqrt_total"] = 1.0 / np.sqrt(grouped["total"] + eps)

# -----------------------------
# Add color/material as one-hot
# -----------------------------
color_dummies = pd.get_dummies(grouped["target_id"], prefix="color")

# -----------------------------
# Select features
# -----------------------------
base_features = [
    "log_total",
    "log_current",
    "log_ambient",
    "R_norm",
    "G_norm",
    "B_norm",
    "R_over_G",
    "R_over_B",
    "G_over_B",
    "log_R_over_G",
    "log_R_over_B",
    "log_G_over_B",
    "inv_sqrt_total"
]

X = pd.concat([grouped[base_features], color_dummies], axis=1)
# y = grouped["distance_mm"]
# y_mm = grouped["distance_mm"]




grouped["log_distance"] = np.log(grouped["distance_mm"] + eps)

y = grouped["log_distance"]
y_mm = grouped["distance_mm"]  # keep for evaluation


# Save feature columns so inference uses same order
feature_columns = X.columns.tolist()

# -----------------------------
# Train/Test split with stratification by distance
# -----------------------------
X_train, X_test, y_train, y_test, y_train_mm, y_test_mm = train_test_split(
    X,
    y,
    y_mm,
    test_size=0.2,
    random_state=42,
    stratify=grouped["distance_mm"]
)

print("Train distances:", sorted(np.unique(y_train_mm)))
print("Test distances:", sorted(np.unique(y_test_mm)))
print("Train size:", len(X_train))
print("Test size:", len(X_test))

# -----------------------------
# XGBoost regressor
# -----------------------------
model = xgb.XGBRegressor(
    n_estimators=2000,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=1,
    gamma=0.00,
    reg_alpha=0.0,
    reg_lambda=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

model.fit(X_train, y_train)

# y_pred_mm = model.predict(X_test)
# y_test_mm = np.array(y_test_mm)

y_pred_log = model.predict(X_test)
y_pred_mm = np.exp(y_pred_log)

# -----------------------------
# Limit evaluation range
# -----------------------------
mask_0_120 = (y_test_mm >= 0) & (y_test_mm <= 120)

y_test_sel = y_test_mm[mask_0_120]
y_pred_sel = y_pred_mm[mask_0_120]

# -----------------------------
# Relative error statistics (0–120 mm)
# -----------------------------
relative_error = np.abs(y_pred_sel - y_test_sel)
percent_error = 1.0 * relative_error

n_total = len(y_test_mm)
n_over_20pct = np.sum(percent_error > 10.0)
frac_over_20pct = n_over_20pct / n_total



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

# -----------------------------
# Save model + feature columns
# -----------------------------
model.save_model("distance_model_with_color.json")

with open("distance_feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

print("Saved model to distance_model_with_color.json")
print("Saved feature columns to distance_feature_columns.pkl")

# -----------------------------
# Plot: True vs Predicted
# -----------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test_mm, y_pred_mm, alpha=0.7)
plt.plot([y_mm.min(), y_mm.max()], [y_mm.min(), y_mm.max()], "r--")
plt.xlabel("True Distance (mm)")
plt.ylabel("Predicted Distance (mm)")
plt.title("XGBoost Distance Model with Color Input")
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

print(f"Points with >20% error: {n_over_20pct} / {n_total}")
print(f"Fraction with >20% error: {frac_over_20pct:.3f}")
print(f"Percentage with >20% error: {100.0 * frac_over_20pct:.2f}%")

plt.figure(figsize=(7, 6))
plt.scatter(y_test_mm, residual, alpha=0.7)
plt.axhline(0, linestyle="--")
plt.xlabel("True Distance (mm)")
plt.ylabel("Residual (Predicted - True) (mm)")
plt.ylim(-75, 75)
plt.xlim(15,135)
plt.title("Residual vs Distance")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Feature importance
# -----------------------------
importance = model.feature_importances_
order = np.argsort(importance)[::-1]

plt.figure(figsize=(12, 6))
plt.bar(np.array(feature_columns)[order], importance[order])
plt.xticks(rotation=60, ha="right")
plt.ylabel("Importance")
plt.title("Feature Importance - Distance Model with Color")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 6))
plt.scatter(y_test_mm, y_pred_mm, alpha=0.7)
plt.plot([y_mm.min(), y_mm.max()], [y_mm.min(), y_mm.max()], "r--")

plt.xscale("log")
plt.yscale("log")

plt.xlabel("True Distance (mm)")
plt.ylabel("Predicted Distance (mm)")
plt.title("Log-Log XGBoost Distance Model")
plt.grid(True, which="both")
plt.tight_layout()
plt.show()