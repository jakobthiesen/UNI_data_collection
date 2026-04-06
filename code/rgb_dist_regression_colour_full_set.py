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

df["target_id"] = df["target_id"].astype(str)

# -----------------------------
# Clean data
# -----------------------------
df = df.dropna(subset=[
    "distance_mm", "current_uA", "ambient", "R", "G", "B", "target_id"
]).copy()

# Optional: exclude black
# df = df[df["target_id"] != "black"].copy()

df["total"] = df["R"] + df["G"] + df["B"]

df = df[
    (df["total"] > 0) &
    (df["current_uA"] > 0) &
    (df["distance_mm"] > 0)
].copy()

# -----------------------------
# Keep only desired distance range
# -----------------------------
df = df[
    (df["distance_mm"] >= 20) &
    (df["distance_mm"] <= 80)
].copy()

# -----------------------------
# Feature engineering on FULL dataset
# -----------------------------
eps = 1e-9

df["log_total"] = np.log(df["total"] + eps)
df["log_current"] = np.log(df["current_uA"] + eps)
df["log_ambient"] = np.log(df["ambient"] + eps)

df["R_norm"] = df["R"] / (df["total"] + eps)
df["G_norm"] = df["G"] / (df["total"] + eps)
df["B_norm"] = df["B"] / (df["total"] + eps)

df["R_over_G"] = df["R"] / (df["G"] + eps)
df["R_over_B"] = df["R"] / (df["B"] + eps)
df["G_over_B"] = df["G"] / (df["B"] + eps)

df["log_R_over_G"] = np.log(df["R_over_G"] + eps)
df["log_R_over_B"] = np.log(df["R_over_B"] + eps)
df["log_G_over_B"] = np.log(df["G_over_B"] + eps)

df["inv_sqrt_total"] = 1.0 / np.sqrt(df["total"] + eps)

df["log_distance"] = np.log(df["distance_mm"] + eps)

# -----------------------------
# Add color/material as one-hot
# -----------------------------
color_dummies = pd.get_dummies(df["target_id"], prefix="color")

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

X = pd.concat([df[base_features], color_dummies], axis=1)
y = df["log_distance"]
y_mm = df["distance_mm"]

feature_columns = X.columns.tolist()

# -----------------------------
# Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test, y_train_mm, y_test_mm = train_test_split(
    X,
    y,
    y_mm,
    test_size=0.2,
    random_state=42,
    stratify=df["distance_mm"]
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

# -----------------------------
# Predictions
# -----------------------------
y_pred_log = model.predict(X_test)
y_pred_mm = np.exp(y_pred_log)
y_test_mm = np.array(y_test_mm)

# -----------------------------
# Limit evaluation range
# -----------------------------
mask_0_120 = (y_test_mm >= 0) & (y_test_mm <= 120)

y_test_sel = y_test_mm[mask_0_120]
y_pred_sel = y_pred_mm[mask_0_120]

# -----------------------------
# Relative / threshold error statistics (0-120 mm)
# -----------------------------
absolute_error_sel = np.abs(y_pred_sel - y_test_sel)

n_total_sel = len(y_test_sel)
n_over_10mm = np.sum(absolute_error_sel > 10.0)
frac_over_10mm = n_over_10mm / n_total_sel if n_total_sel > 0 else np.nan

# If you want actual percent error instead, use:
# relative_error_sel = np.abs(y_pred_sel - y_test_sel) / (np.abs(y_test_sel) + 1e-9)
# n_over_10pct = np.sum(relative_error_sel > 0.10)

# -----------------------------
# Evaluation in mm
# -----------------------------
mae = mean_absolute_error(y_test_mm, y_pred_mm)
rmse = np.sqrt(mean_squared_error(y_test_mm, y_pred_mm))

print(f"Number of full samples: {len(df)}")
print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"MAE:  {mae:.2f} mm")
print(f"RMSE: {rmse:.2f} mm")

print(f"Points with >10 mm error in 0-120 mm: {n_over_10mm} / {n_total_sel}")
print(f"Fraction with >10 mm error in 0-120 mm: {frac_over_10mm:.3f}")
print(f"Percentage with >10 mm error in 0-120 mm: {100.0 * frac_over_10mm:.2f}%")

# -----------------------------
# Save model + feature columns
# -----------------------------
model.save_model("distance_model_full_dataset.json")

with open("distance_feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

print("Saved model to distance_model_full_dataset.json")
print("Saved feature columns to distance_feature_columns.pkl")

# -----------------------------
# Plot: True vs Predicted
# -----------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test_mm, y_pred_mm, alpha=0.4)
plt.plot([y_mm.min(), y_mm.max()], [y_mm.min(), y_mm.max()], "r--")
plt.xlabel("True Distance (mm)")
plt.ylabel("Predicted Distance (mm)")
plt.title("XGBoost Distance Model with Color Input (Full Dataset)")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Plot: Absolute Error vs Distance
# -----------------------------
error = np.abs(y_test_mm - y_pred_mm)

plt.figure(figsize=(7, 6))
plt.scatter(y_test_mm, error, alpha=0.4)
plt.xlabel("Distance (mm)")
plt.ylabel("Absolute Error (mm)")
plt.title("Absolute Error vs Distance (Full Dataset)")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Plot: Residual vs Distance
# -----------------------------
residual = y_pred_mm - y_test_mm

plt.figure(figsize=(7, 6))
plt.scatter(y_test_mm, residual, alpha=0.4)
plt.axhline(0, linestyle="--")
plt.xlabel("True Distance (mm)")
plt.ylabel("Residual (Predicted - True) (mm)")
plt.ylim(-75, 75)
plt.xlim(15, 205)
plt.title("Residual vs Distance (Full Dataset)")
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
plt.title("Feature Importance - Distance Model with Color (Full Dataset)")
plt.tight_layout()
plt.show()

# -----------------------------
# Plot: True vs Predicted on log-log axes
# -----------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test_mm, y_pred_mm, alpha=0.4)
plt.plot([y_mm.min(), y_mm.max()], [y_mm.min(), y_mm.max()], "r--")

plt.xscale("log")
plt.yscale("log")

plt.xlabel("True Distance (mm)")
plt.ylabel("Predicted Distance (mm)")
plt.title("Log-Log XGBoost Distance Model (Full Dataset)")
plt.grid(True, which="both")
plt.tight_layout()
plt.show()