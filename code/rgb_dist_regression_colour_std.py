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

agg["u_ambient"] = agg["ambient_std"] / np.sqrt(agg["ambient_n"].clip(lower=1))
agg["u_R"] = agg["R_std"] / np.sqrt(agg["R_n"].clip(lower=1))
agg["u_G"] = agg["G_std"] / np.sqrt(agg["G_n"].clip(lower=1))
agg["u_B"] = agg["B_std"] / np.sqrt(agg["B_n"].clip(lower=1))

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

# Optional: train only on 20 mm to 140 mm
agg = agg[
    (agg["distance_mm"] >= 20) &
    (agg["distance_mm"] <= 160)
].copy()

# -----------------------------
# Feature engineering
# -----------------------------
agg["log_total"] = np.log(agg["total"] + eps)
agg["log_current"] = np.log(agg["current_uA"] + eps)
agg["log_ambient"] = np.log(agg["ambient"] + eps)
agg["log_distance"] = np.log(agg["distance_mm"] + eps)

# Spectral features
agg["R_norm"] = agg["R"] / (agg["total"] + eps)
agg["G_norm"] = agg["G"] / (agg["total"] + eps)
agg["B_norm"] = agg["B"] / (agg["total"] + eps)

agg["R_over_G"] = agg["R"] / (agg["G"] + eps)
agg["R_over_B"] = agg["R"] / (agg["B"] + eps)
agg["G_over_B"] = agg["G"] / (agg["B"] + eps)

agg["log_R_over_G"] = np.log(agg["R_over_G"] + eps)
agg["log_R_over_B"] = np.log(agg["R_over_B"] + eps)
agg["log_G_over_B"] = np.log(agg["G_over_B"] + eps)

# -----------------------------
# Uncertainty propagation
# -----------------------------
agg["u_total"] = np.sqrt(agg["u_R"]**2 + agg["u_G"]**2 + agg["u_B"]**2)

agg["u_log_total"] = agg["u_total"] / (agg["total"] + eps)
agg["u_log_ambient"] = agg["u_ambient"] / (agg["ambient"] + eps)

agg["u_R_norm"] = agg["R_norm"] * np.sqrt(
    (agg["u_R"] / (agg["R"] + eps))**2 +
    (agg["u_total"] / (agg["total"] + eps))**2
)

agg["u_G_norm"] = agg["G_norm"] * np.sqrt(
    (agg["u_G"] / (agg["G"] + eps))**2 +
    (agg["u_total"] / (agg["total"] + eps))**2
)

agg["u_B_norm"] = agg["B_norm"] * np.sqrt(
    (agg["u_B"] / (agg["B"] + eps))**2 +
    (agg["u_total"] / (agg["total"] + eps))**2
)

agg["u_R_over_G"] = agg["R_over_G"] * np.sqrt(
    (agg["u_R"] / (agg["R"] + eps))**2 +
    (agg["u_G"] / (agg["G"] + eps))**2
)

agg["u_R_over_B"] = agg["R_over_B"] * np.sqrt(
    (agg["u_R"] / (agg["R"] + eps))**2 +
    (agg["u_B"] / (agg["B"] + eps))**2
)

agg["u_G_over_B"] = agg["G_over_B"] * np.sqrt(
    (agg["u_G"] / (agg["G"] + eps))**2 +
    (agg["u_B"] / (agg["B"] + eps))**2
)

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

# -----------------------------
# Color one-hot encoding
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
    "log_G_over_B"
]

uncertainty_features = [
    "u_total",
    "u_log_total",
    "u_log_ambient",
    "u_R_norm",
    "u_G_norm",
    "u_B_norm",
    "u_R_over_G",
    "u_R_over_B",
    "u_G_over_B",
    "u_log_R_over_G",
    "u_log_R_over_B",
    "u_log_G_over_B"
]

X = pd.concat([agg[base_features + uncertainty_features], color_dummies], axis=1)
y = agg["log_distance"]
y_mm = agg["distance_mm"]

feature_columns = X.columns.tolist()

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test, y_train_mm, y_test_mm = train_test_split(
    X, y, y_mm,
    test_size=0.2,
    random_state=42,
    stratify=agg["distance_mm"]
)

# -----------------------------
# XGBoost regressor
# -----------------------------
model = xgb.XGBRegressor(
    n_estimators=10000,
    max_depth=2,
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

# -----------------------------
# Save model + feature columns
# -----------------------------
model.save_model("distance_model_with_color_uncertainty.json")

with open("distance_feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

print("Saved model to distance_model_with_color_uncertainty.json")
print("Saved feature columns to distance_feature_columns.pkl")

# -----------------------------
# Plot: True vs Predicted
# -----------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test_mm, y_pred_mm, alpha=0.7)
plt.plot([y_mm.min(), y_mm.max()], [y_mm.min(), y_mm.max()], "r--")
plt.xlabel("True Distance (mm)")
plt.ylabel("Predicted Distance (mm)")
plt.title("XGBoost Distance Model with Color + Uncertainty")
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
plt.ylim(-100,100)
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
plt.title("Feature Importance - Distance Model with Color + Uncertainty")
plt.tight_layout()
plt.show()