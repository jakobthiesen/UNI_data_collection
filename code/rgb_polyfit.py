import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def estimate_distance(signal, a2, a1, a0):
    y = np.log(signal)

    A = a2
    B = a1
    C = a0 - y

    d = (-B + np.sqrt(B**2 - 4*A*C)) / (2*A)
    return d

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
# df["total"] = df["G"] / df["B"]

# Group
grouped = df.groupby(
    ["target_id", "distance_mm", "current_uA"],
    as_index=False
).mean(numeric_only=True)


# plt.figure(figsize=(10, 6))

for color in grouped["target_id"].unique():
    subset = grouped[
        (grouped["target_id"] == color) &
        (grouped["current_uA"] == 8000)
    ].sort_values("distance_mm")

    # plt.plot(
    #     subset["distance_mm"],
    #     subset["total"],
    #     marker='o',
    #     label=color
    # )

subset = grouped[
    (grouped["target_id"] == "yellow") &
    (grouped["current_uA"] == 8000)
].sort_values("distance_mm")


x = np.log(subset["distance_mm"].values)
y = np.log(subset["total"].values)

coeffs = np.polyfit(x, y, 1)
y_fit = np.polyval(coeffs, x)

print(coeffs)

plt.figure()
plt.plot(x, y, 'o', label="data")
plt.plot(x, y_fit, '-', label="quadratic fit")
plt.xlabel("log(Distance) (mm)")
plt.ylabel("log(signal)")
plt.legend()
plt.grid()
plt.show()



# plt.xlabel("Distance (mm)")
# plt.ylabel("Total Signal")
# plt.title("Signal vs Distance (8 mA)")
# # plt.yscale("log")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()