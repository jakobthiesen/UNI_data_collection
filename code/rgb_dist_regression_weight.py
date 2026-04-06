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

# # 👉 EXCLUDE BLACK HERE
df = df[df["target_id"] != "black"].copy()

df["total"] = df["R"] + df["G"] + df["B"]

df = df[
    (df["total"] > 0) &
    (df["current_uA"] > 0) &
    (df["distance_mm"] > 0)
].copy()

# -----------------------------
# Aggregate mean + std per repeat-condition
# -----------------------------
group_cols = ["session_id", "repeat_id", "target_id", "distance_mm", "current_uA"]

agg = df.groupby(group_cols).agg(
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
    agg[col] = agg[col].fillna(0.0)

# -----------------------------
# Standard uncertainty of the mean
# -----------------------------
eps = 1e-9


# -----------------------------
# Current uncertainty
# -----------------------------
# Your stated current uncertainty is 0.040 mA.
# Since the column is current_uA, convert to microamps:
CURRENT_UNCERTAINTY_UA = 40.0   # 0.040 mA = 40 µA

# Optional: if your current column is actually in mA, use:
# CURRENT_UNCERTAINTY_UA = 0.040

agg["u_current"] = CURRENT_UNCERTAINTY_UA

# -----------------------------
# Derived means
# -----------------------------
agg["ambient"] = agg["ambient_mean"]
agg["R"] = agg["R_mean"]
agg["G"] = agg["G_mean"]
agg["B"] = agg["B_mean"]
agg["total"] = agg["R"] + agg["G"] + agg["B"]

agg = agg[
    (agg["total"] > 0) &
    (agg["current_uA"] > 0) &
    (agg["distance_mm"] > 0)
].copy()

# -----------------------------
# Optional training-range filter
# -----------------------------
agg = agg[
    (agg["distance_mm"] >= 20) &
    (agg["distance_mm"] <= 80)
].copy()

# -----------------------------
# Uncertainty propagation
# -----------------------------
agg["u_ambient"] = agg["ambient_std"] / np.sqrt(agg["ambient_n"].clip(lower=1))
agg["u_R"] = (agg["R_std"] / np.sqrt(agg["R_n"].clip(lower=1)))
agg["u_G"] = agg["G_std"] / np.sqrt(agg["G_n"].clip(lower=1))
agg["u_B"] = agg["B_std"] / np.sqrt(agg["B_n"].clip(lower=1))



agg["u_total"] = np.sqrt(agg["u_R"]**2 + agg["u_G"]**2 + agg["u_B"]**2 + agg["u_ambient"]**2)
agg["u_log_total"] = agg["u_total"] / (agg["total"] + eps)
agg["u_log_ambient"] = agg["u_ambient"] / (agg["ambient"] + eps)
agg["u_log_current"] = agg["u_current"] / (agg["current_uA"] + eps)

agg["u_log_R_over_G"] = np.sqrt(
    (agg["u_R"] / (agg["R"] + eps))**2 +
    (agg["u_G"] / (agg["G"] + eps))**2
)

agg["u_log_R_over_B"] = np.sqrt(
    (agg["u_R"] / (agg["R"] + eps))**2 +
    (agg["u_B"] / (agg["B"] + eps))**2
)

agg["u_log_G_over_B"] = np.sqrt(
    (agg["u_G"] / (agg["G"] + eps))**2 +
    (agg["u_B"] / (agg["B"] + eps))**2
)

agg["u_R_inl"] = 4.0/agg["R_mean"]
agg["u_G_inl"] = 4.0/agg["G_mean"]
agg["u_B_inl"] = 4.0/agg["B_mean"]
agg["u_amb_inl"] = 4.0/agg["ambient_mean"]

agg["u_inl"] = np.sqrt(agg["u_R_inl"]**2 + agg["u_G_inl"]**2 +agg["u_B_inl"]**2 +agg["u_amb_inl"]**2)
# agg["u_inl"] = 50.0 / agg["total"]

# agg["u_adc_noise"] = 11.0 / agg["total"]

# -----------------------------
# Build scalar sample uncertainty
# -----------------------------
agg["u_sample"] = np.sqrt(
    agg["u_log_total"]**2 +
    agg["u_log_current"]**2 +   # <-- added current contribution
    agg["u_log_ambient"]**2+
    agg["u_log_R_over_G"]**2 +
    agg["u_log_R_over_B"]**2 +
    agg["u_log_G_over_B"]**2 +
    agg["u_inl"]**2 
    # agg["u_adc_noise"]**2
)

u = agg["u_sample"].to_numpy()

# Choose transition location from data
u0 = np.percentile(u, 70)   # start downweighting around upper-middle uncertainty
k = 12.0                    # steepness
w_min = 0.01                # minimum weight floor

agg["sample_weight"] = w_min + (1.0 - w_min) / (1.0 + np.exp(k * (u - u0)))

# Optional normalization
agg["sample_weight"] = agg["sample_weight"] / agg["sample_weight"].mean()

# agg["sample_weight"] = 1.0 / (agg["u_sample"]**2 + 1e-9)
# agg["sample_weight"] = agg["sample_weight"] / agg["sample_weight"].mean()
# agg["sample_weight"] = agg["sample_weight"].clip(lower=0.001, upper=10.0)



# -----------------------------
# Feature engineering
# -----------------------------
agg["log_total"] = np.log(agg["total"] + eps)
agg["log_current"] = np.log(agg["current_uA"] + eps)
agg["log_ambient"] = np.log(agg["ambient"] + eps)
agg["log_distance"] = np.log(agg["distance_mm"] + eps)

agg["R_norm"] = agg["R"] / (agg["total"] + eps)
agg["G_norm"] = agg["G"] / (agg["total"] + eps)
agg["B_norm"] = agg["B"] / (agg["total"] + eps)

agg["R_over_G"] = agg["R"] / (agg["G"] + eps)
agg["R_over_B"] = agg["R"] / (agg["B"] + eps)
agg["G_over_B"] = agg["G"] / (agg["B"] + eps)

agg["log_R_over_G"] = np.log(agg["R_over_G"] + eps)
agg["log_R_over_B"] = np.log(agg["R_over_B"] + eps)
agg["log_G_over_B"] = np.log(agg["G_over_B"] + eps)

agg["inv_sqrt_total"] = 1.0 / np.sqrt(agg["total"]+eps)



# -----------------------------
# Optional color/material input
# -----------------------------
color_dummies = pd.get_dummies(agg["target_id"], prefix="color")

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

X = pd.concat([agg[base_features], color_dummies], axis=1)
y = agg["log_distance"]
# y = agg["distance_mm"]
y_mm = agg["distance_mm"]
w = agg["sample_weight"]

feature_columns = X.columns.tolist()

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test, y_train_mm, y_test_mm, w_train, w_test = train_test_split(
    X, y, y_mm, w,
    test_size=0.2,
    random_state=42,
    stratify=agg["distance_mm"]
)
# X_train, X_test, y_train, y_test, y_train_mm, y_test_mm, w_train, w_test = train_test_split(
#     X, y, y_mm, w,
#     test_size=0.2,
#     random_state=42,
#     stratify=agg["distance_mm"]
# )

# -----------------------------
# Train/Test split by distance range
# -----------------------------
# train_mask = (agg["distance_mm"] >= 20) & (agg["distance_mm"] < 60)
# test_mask  = (agg["distance_mm"] >= 80) & (agg["distance_mm"] <= 120)

# X_train = X[train_mask]
# y_train = y[train_mask]
# y_train_mm = y_mm[train_mask]
# w_train = w[train_mask]

# X_test = X[test_mask]
# y_test = y[test_mask]
# y_test_mm = y_mm[test_mask]
# w_test = w[test_mask]

# print("Train distances:", sorted(np.unique(y_train_mm)))
# print("Test distances:", sorted(np.unique(y_test_mm)))


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
# y_pred_mm = model.predict(X_test)
y_test_mm = np.array(y_test_mm)


# -----------------------------
# Limit evaluation range
# -----------------------------
mask_0_120 = (y_test_mm >= 0) & (y_test_mm <= 120)

y_test_sel = y_test_mm[mask_0_120]
y_pred_sel = y_pred_mm[mask_0_120]

# -----------------------------
# Relative error statistics (0–120 mm)
# -----------------------------
# relative_error = np.abs(y_pred_sel - y_test_sel) / (np.abs(y_test_sel) + 1e-9)

relative_error = np.abs(y_pred_sel - y_test_sel)
percent_error = 1.0 * relative_error

n_total = len(y_test_mm)
n_over_20pct = np.sum(percent_error > 10.0)
frac_over_20pct = n_over_20pct / n_total

# -----------------------------
# Evaluation
# -----------------------------
mae = mean_absolute_error(y_test_mm, y_pred_mm)
rmse = np.sqrt(mean_squared_error(y_test_mm, y_pred_mm))

print(f"Number of grouped samples: {len(agg)}")
print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"MAE:  {mae:.2f} mm")
print(f"RMSE: {rmse:.2f} mm")

print("Train distances:", sorted(np.unique(y_train_mm)))
print("Test distances:", sorted(np.unique(y_test_mm)))

print("\nWeight statistics:")
print(agg["sample_weight"].describe())

print("\nCurrent uncertainty settings:")
print(f"Current uncertainty: {CURRENT_UNCERTAINTY_UA:.3f} µA")
print("Allowed current steps (mA): 0.5, 1, 2, 4, 8")

# -----------------------------
# Save model + feature columns
# -----------------------------
model.save_model("distance_model_weighted_uncertainty.json")

with open("distance_feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

print("Saved model to distance_model_weighted_uncertainty.json")
print("Saved feature columns to distance_feature_columns.pkl")

# -----------------------------
# Plot: True vs Predicted
# -----------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test_mm, y_pred_mm, alpha=0.7)
plt.plot([y_mm.min(), y_mm.max()], [y_mm.min(), y_mm.max()], "r--")
plt.xlabel("True Distance (mm)")
plt.ylabel("Predicted Distance (mm)")
plt.title("Weighted-Uncertainty XGBoost Distance Model")
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

print(f"MAE:  {mae:.2f} mm")
print(f"RMSE: {rmse:.2f} mm")

print(f"Points with >20% error: {n_over_20pct} / {n_total}")
print(f"Fraction with >20% error: {frac_over_20pct:.3f}")
print(f"Percentage with >20% error: {100.0 * frac_over_20pct:.2f}%")


# -----------------------------
# Plot: Residual vs Distance
# -----------------------------
residual = y_pred_mm - y_test_mm

plt.figure(figsize=(7, 6))
plt.scatter(y_test_mm, residual, alpha=0.7)
plt.axhline(0, linestyle="--")
plt.xlabel("True Distance (mm)")
plt.ylabel("Residual (Predicted - True) (mm)")
plt.title("Residual vs Distance (Weighted)")
plt.ylim(-75, 75)
plt.xlim(15,135)
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
plt.title("Feature Importance - Weighted-Uncertainty Distance Model")
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