import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle

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

df["target_id"] = df["target_id"].astype(str)

# -----------------------------
# Clean data
# -----------------------------
df = df.dropna(subset=[
    "distance_mm", "current_uA", "ambient", "R", "G", "B", "target_id"
]).copy()

# Optional:
# df = df[df["target_id"] != "black"].copy()

df["total"] = df["R"] + df["G"] + df["B"]

df = df[
    (df["total"] > 0) &
    (df["current_uA"] > 0) &
    (df["distance_mm"] > 0)
].copy()

# -----------------------------
# Limit range
# -----------------------------
df = df[
    (df["distance_mm"] >= 20) &
    (df["distance_mm"] <= 80)
].copy()

# -----------------------------
# Estimate uncertainty from 25-sample repeat-condition batches
# -----------------------------
group_cols = ["session_id", "repeat_id", "target_id", "distance_mm", "current_uA"]

batch_stats = df.groupby(group_cols).agg(
    ambient_mean=("ambient", "mean"),
    ambient_std=("ambient", "std"),
    ambient_n=("ambient", "count"),

    R_mean=("R", "mean"),
    R_std=("R", "std"),
    R_n=("R", "count"),

    G_mean=("G", "mean"),
    G_std=("G", "std"),
    G_n=("G", "count"),

    B_mean=("B", "mean"),
    B_std=("B", "std"),
    B_n=("B", "count")
).reset_index()

for col in ["ambient_std", "R_std", "G_std", "B_std"]:
    batch_stats[col] = batch_stats[col].fillna(0.0)

eps = 1e-9

# -----------------------------
# Current uncertainty
# -----------------------------
CURRENT_UNCERTAINTY_UA = 40.0
batch_stats["u_current"] = CURRENT_UNCERTAINTY_UA

# -----------------------------
# Derived means for uncertainty propagation
# -----------------------------
batch_stats["ambient"] = batch_stats["ambient_mean"]
batch_stats["R"] = batch_stats["R_mean"]
batch_stats["G"] = batch_stats["G_mean"]
batch_stats["B"] = batch_stats["B_mean"]
batch_stats["total"] = batch_stats["R"] + batch_stats["G"] + batch_stats["B"]

# -----------------------------
# Standard uncertainty of the mean from each 25-sample batch
# -----------------------------
batch_stats["u_ambient"] = batch_stats["ambient_std"] / np.sqrt(batch_stats["ambient_n"].clip(lower=1))
batch_stats["u_R"] = batch_stats["R_std"] / np.sqrt(batch_stats["R_n"].clip(lower=1))
batch_stats["u_G"] = batch_stats["G_std"] / np.sqrt(batch_stats["G_n"].clip(lower=1))
batch_stats["u_B"] = batch_stats["B_std"] / np.sqrt(batch_stats["B_n"].clip(lower=1))

# -----------------------------
# Propagate uncertainty in log-domain quantities
# -----------------------------
batch_stats["u_total"] = np.sqrt(
    batch_stats["u_R"]**2 +
    batch_stats["u_G"]**2 +
    batch_stats["u_B"]**2 +
    batch_stats["u_ambient"]**2
)

batch_stats["u_log_total"] = batch_stats["u_total"] / (batch_stats["total"] + eps)
batch_stats["u_log_ambient"] = batch_stats["u_ambient"] / (batch_stats["ambient"] + eps)
batch_stats["u_log_current"] = batch_stats["u_current"] / (batch_stats["current_uA"] + eps)

batch_stats["u_log_R_over_G"] = np.sqrt(
    (batch_stats["u_R"] / (batch_stats["R"] + eps))**2 +
    (batch_stats["u_G"] / (batch_stats["G"] + eps))**2
)

batch_stats["u_log_R_over_B"] = np.sqrt(
    (batch_stats["u_R"] / (batch_stats["R"] + eps))**2 +
    (batch_stats["u_B"] / (batch_stats["B"] + eps))**2
)

batch_stats["u_log_G_over_B"] = np.sqrt(
    (batch_stats["u_G"] / (batch_stats["G"] + eps))**2 +
    (batch_stats["u_B"] / (batch_stats["B"] + eps))**2
)

# -----------------------------
# Instrument / INL-like term
# -----------------------------
batch_stats["u_R_inl"] = 4.0 / (batch_stats["R_mean"] + eps)
batch_stats["u_G_inl"] = 4.0 / (batch_stats["G_mean"] + eps)
batch_stats["u_B_inl"] = 4.0 / (batch_stats["B_mean"] + eps)
batch_stats["u_amb_inl"] = 4.0 / (batch_stats["ambient_mean"] + eps)

batch_stats["u_inl"] = np.sqrt(
    batch_stats["u_R_inl"]**2 +
    batch_stats["u_G_inl"]**2 +
    batch_stats["u_B_inl"]**2 +
    batch_stats["u_amb_inl"]**2
)

# -----------------------------
# Scalar uncertainty per 25-sample batch
# -----------------------------
batch_stats["u_sample"] = np.sqrt(
    batch_stats["u_log_total"]**2 +
    batch_stats["u_log_current"]**2 +
    batch_stats["u_log_ambient"]**2 +
    batch_stats["u_log_R_over_G"]**2 +
    batch_stats["u_log_R_over_B"]**2 +
    batch_stats["u_log_G_over_B"]**2 +
    batch_stats["u_inl"]**2
)

# -----------------------------
# Smooth uncertainty -> weight map
# -----------------------------
u = batch_stats["u_sample"].to_numpy()
u0 = np.percentile(u, 70)
k = 12.0
w_min = 0.01

batch_stats["sample_weight"] = w_min + (1.0 - w_min) / (1.0 + np.exp(k * (u - u0)))
batch_stats["sample_weight"] = batch_stats["sample_weight"] / batch_stats["sample_weight"].mean()

# -----------------------------
# Merge batch uncertainty/weights back to FULL raw dataset
# -----------------------------
merge_cols = group_cols + [
    "u_ambient",
    "u_R",
    "u_G",
    "u_B",
    "u_total",
    "u_log_total",
    "u_log_ambient",
    "u_log_current",
    "u_log_R_over_G",
    "u_log_R_over_B",
    "u_log_G_over_B",
    "u_inl",
    "u_sample",
    "sample_weight"
]

df_full = df.merge(
    batch_stats[merge_cols],
    on=group_cols,
    how="left"
)

# -----------------------------
# Feature engineering on FULL raw rows
# -----------------------------
df_full["log_total"] = np.log(df_full["total"] + eps)
df_full["log_current"] = np.log(df_full["current_uA"] + eps)
df_full["log_ambient"] = np.log(df_full["ambient"] + eps)
df_full["log_distance"] = np.log(df_full["distance_mm"] + eps)

df_full["R_norm"] = df_full["R"] / (df_full["total"] + eps)
df_full["G_norm"] = df_full["G"] / (df_full["total"] + eps)
df_full["B_norm"] = df_full["B"] / (df_full["total"] + eps)

df_full["R_over_G"] = df_full["R"] / (df_full["G"] + eps)
df_full["R_over_B"] = df_full["R"] / (df_full["B"] + eps)
df_full["G_over_B"] = df_full["G"] / (df_full["B"] + eps)

df_full["log_R_over_G"] = np.log(df_full["R_over_G"] + eps)
df_full["log_R_over_B"] = np.log(df_full["R_over_B"] + eps)
df_full["log_G_over_B"] = np.log(df_full["G_over_B"] + eps)

df_full["inv_sqrt_total"] = 1.0 / np.sqrt(df_full["total"] + eps)

# -----------------------------
# Optional color/material input
# -----------------------------
color_dummies = pd.get_dummies(df_full["target_id"], prefix="color")

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

X = pd.concat([df_full[base_features], color_dummies], axis=1)
y = df_full["log_distance"]
y_mm = df_full["distance_mm"]
w = df_full["sample_weight"]

feature_columns = X.columns.tolist()

# -----------------------------
# Train/test split
# NOTE: still row-wise split; group-aware split would be better
# -----------------------------
X_train, X_test, y_train, y_test, y_train_mm, y_test_mm, w_train, w_test = train_test_split(
    X, y, y_mm, w,
    test_size=0.2,
    random_state=42,
    stratify=df_full["distance_mm"]
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

model.fit(X_train, y_train, sample_weight=w_train)

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
# Error statistics in 0-120 mm
# -----------------------------
absolute_error_sel = np.abs(y_pred_sel - y_test_sel)

n_total_sel = len(y_test_sel)
n_over_10mm = np.sum(absolute_error_sel > 10.0)
frac_over_10mm = n_over_10mm / n_total_sel if n_total_sel > 0 else np.nan

# -----------------------------
# Evaluation
# -----------------------------
mae = mean_absolute_error(y_test_mm, y_pred_mm)
rmse = np.sqrt(mean_squared_error(y_test_mm, y_pred_mm))

print(f"Number of raw samples: {len(df_full)}")
print(f"Number of uncertainty batches: {len(batch_stats)}")
print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"MAE:  {mae:.2f} mm")
print(f"RMSE: {rmse:.2f} mm")

print("\nWeight statistics:")
print(df_full["sample_weight"].describe())

print("\nCurrent uncertainty settings:")
print(f"Current uncertainty: {CURRENT_UNCERTAINTY_UA:.3f} µA")
print("Allowed current steps (mA): 0.5, 1, 2, 4, 8")

print(f"\nPoints with >10 mm error in 0-120 mm: {n_over_10mm} / {n_total_sel}")
print(f"Fraction with >10 mm error in 0-120 mm: {frac_over_10mm:.3f}")
print(f"Percentage with >10 mm error in 0-120 mm: {100.0 * frac_over_10mm:.2f}%")

# -----------------------------
# Save model + feature columns
# -----------------------------
model.save_model("distance_model_full_weighted_from_batch_uncertainty.json")

with open("distance_feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

print("Saved model to distance_model_full_weighted_from_batch_uncertainty.json")
print("Saved feature columns to distance_feature_columns.pkl")

# -----------------------------
# Plot: True vs Predicted
# -----------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test_mm, y_pred_mm, alpha=0.35)
plt.plot([y_mm.min(), y_mm.max()], [y_mm.min(), y_mm.max()], "r--")
plt.xlabel("True Distance (mm)")
plt.ylabel("Predicted Distance (mm)")
plt.title("Full Dataset Model, Batch-Derived Uncertainty Weights")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Plot: Absolute Error vs Distance
# -----------------------------
error = np.abs(y_test_mm - y_pred_mm)

plt.figure(figsize=(7, 6))
plt.scatter(y_test_mm, error, alpha=0.35)
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
plt.scatter(y_test_mm, residual, alpha=0.35)
plt.axhline(0, linestyle="--")
plt.xlabel("True Distance (mm)")
plt.ylabel("Residual (Predicted - True) (mm)")
plt.title("Residual vs Distance")
plt.ylim(-75, 75)
plt.xlim(15, 205)
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Feature importance
# -----------------------------
importance = model.feature_importances_
order = np.argsort(importance)[::-1]

plt.figure(figsize=(14, 6))
plt.bar(np.array(feature_columns)[order], importance[order])
plt.xticks(rotation=70, ha="right")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# -----------------------------
# Log-log plot
# -----------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test_mm, y_pred_mm, alpha=0.35)
plt.plot([y_mm.min(), y_mm.max()], [y_mm.min(), y_mm.max()], "r--")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("True Distance (mm)")
plt.ylabel("Predicted Distance (mm)")
plt.title("Log-Log XGBoost Distance Model")
plt.grid(True, which="both")
plt.tight_layout()
plt.show()