import pandas as pd
import matplotlib.pyplot as plt

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

# Force numeric
for col in ["ambient", "R", "G", "B", "total"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Recompute total (safe)
df["total"] = df["R"] + df["G"] + df["B"]

# Group
grouped = df.groupby(
    ["target_id", "distance_mm", "current_uA"],
    as_index=False
).mean(numeric_only=True)

# Plot
plt.figure(figsize=(10, 6))

for color in grouped["target_id"].unique():
    subset = grouped[
        (grouped["target_id"] == color) &
        (grouped["current_uA"] == 8000)
    ].sort_values("distance_mm")

    plt.loglog(
        subset["distance_mm"],
        subset["total"],
        marker='o',
        label=color
    )

plt.xlabel("Distance (mm)")
plt.ylabel("Total Signal")
plt.title("Signal vs Distance (8 mA)")
# plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()