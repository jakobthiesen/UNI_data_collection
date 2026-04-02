import struct
import time
import serial
import numpy as np
import xgboost as xgb
import pandas as pd
import pickle

PORT = "COM6"
BAUDRATE = 115200
TIMEOUT_S = 5.0

END_CHAR = b"\n"
REPLY_LEN = 17  # 16 bytes float data + '\n'


def open_serial(port: str, baudrate: int = BAUDRATE, timeout: float = TIMEOUT_S) -> serial.Serial:
    ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
    time.sleep(2.0)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    return ser


def read_exactly(ser: serial.Serial, nbytes: int) -> bytes:
    data = bytearray()
    while len(data) < nbytes:
        chunk = ser.read(nbytes - len(data))
        if not chunk:
            raise TimeoutError(f"Timed out waiting for {nbytes} bytes, got {len(data)}")
        data.extend(chunk)
    return bytes(data)


def trigger_capture(current, ser: serial.Serial):
    ser.reset_input_buffer()
    ser.write(f"r{current}\n".encode())
    ser.flush()

    packet = read_exactly(ser, REPLY_LEN)

    if packet[-1:] != END_CHAR:
        raise ValueError("Bad packet")

    return struct.unpack("<ffff", packet[:16])


def capture_series(ser, current, n_samples=10):
    amb_vals = []
    r_vals = []
    g_vals = []
    b_vals = []

    for i in range(n_samples):
        amb, r, g, b = trigger_capture(current, ser)

        amb_vals.append(amb)
        r_vals.append(r)
        g_vals.append(g)
        b_vals.append(b)

        print(
            f"{i+1:3d}/{n_samples} | I={current:8.3f}, "
            f"Amb={amb:8.3f}, R={r:8.3f}, G={g:8.3f}, B={b:8.3f}"
        )

    return amb_vals, r_vals, g_vals, b_vals


def build_distance_features(
    log_total,
    log_current,
    log_ambient,
    R_norm,
    G_norm,
    B_norm,
    R_over_G,
    R_over_B,
    G_over_B,
    log_R_over_G,
    log_R_over_B,
    log_G_over_B,
    color_label,
    distance_feature_columns
):
    row = {
        "log_total": log_total,
        "log_current": log_current,
        "log_ambient": log_ambient,
        "R_norm": R_norm,
        "G_norm": G_norm,
        "B_norm": B_norm,
        "R_over_G": R_over_G,
        "R_over_B": R_over_B,
        "G_over_B": G_over_B,
        "log_R_over_G": log_R_over_G,
        "log_R_over_B": log_R_over_B,
        "log_G_over_B": log_G_over_B,
    }

    # Add all color dummy columns expected by the distance model
    for col in distance_feature_columns:
        if col.startswith("color_"):
            row[col] = 0

    chosen_color_col = f"color_{color_label}"
    if chosen_color_col in distance_feature_columns:
        row[chosen_color_col] = 1
    else:
        raise ValueError(
            f"Predicted color '{color_label}' does not match any color column in distance model."
        )

    features_dist = pd.DataFrame([row])

    # Ensure all required columns exist
    for col in distance_feature_columns:
        if col not in features_dist.columns:
            features_dist[col] = 0

    # Exact order expected by the model
    features_dist = features_dist[distance_feature_columns]
    return features_dist


def main():
    ser = open_serial(PORT)

    try:
        eps = 1e-9

        # -----------------------------
        # Load models
        # -----------------------------
        model_cls = xgb.XGBClassifier()
        model_reg = xgb.XGBRegressor()

        # model_cls.load_model("color_classifier_uncertainty.json")
        model_cls.load_model("color_classifier_weighted_uncertainty.json")
        model_reg.load_model("distance_model_weighted_uncertainty.json")

        cls_feature_columns = [
            # "log_total",
            # "log_current",
            # # "log_ambient",
            # "R_norm",
            # "G_norm",
            # "B_norm",
            # "R_over_G",
            # "R_over_B",
            # "G_over_B",
            # "log_R_over_G",
            # "log_R_over_B",
            # "log_G_over_B",
            # "u_total",
            # "u_log_total",
            # # "u_log_ambient",
            # "u_R_norm",
            # "u_G_norm",
            # "u_B_norm",
            # "u_R_over_G",
            # "u_R_over_B",
            # "u_G_over_B",
            # "u_log_R_over_G",
            # "u_log_R_over_B",
            # "u_log_G_over_B"
            "log_total",
            "log_current",
            # "log_ambient",
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


        with open("label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)

        # Classifier feature list (24 features)
        with open("feature_columns.pkl", "rb") as f:
            classifier_feature_columns = pickle.load(f)

        # Distance regressor feature list (base features + one-hot color columns)
        with open("distance_feature_columns.pkl", "rb") as f:
            distance_feature_columns = pickle.load(f)

        current = 203
        current_uA = current * 39.2157

        while True:
            input(">> Press ENTER to measure...")

            amb_vals, r_vals, g_vals, b_vals = capture_series(ser, current, 25)

            amb_vals = np.array(amb_vals, dtype=float)
            r_vals = np.array(r_vals, dtype=float)
            g_vals = np.array(g_vals, dtype=float)
            b_vals = np.array(b_vals, dtype=float)

            N = len(r_vals)

            amb_mean = np.mean(amb_vals)
            R = np.mean(r_vals)
            G = np.mean(g_vals)
            B = np.mean(b_vals)

            # Standard uncertainty of the mean
            u_R = np.std(r_vals, ddof=1) / np.sqrt(N) if N > 1 else 0.0
            u_G = np.std(g_vals, ddof=1) / np.sqrt(N) if N > 1 else 0.0
            u_B = np.std(b_vals, ddof=1) / np.sqrt(N) if N > 1 else 0.0
            u_amb = np.std(amb_vals, ddof=1) / np.sqrt(N) if N > 1 else 0.0

            T = R + G + B
            if T <= 0:
                print("Invalid measurement: total signal <= 0")
                continue

            u_total = np.sqrt(u_R**2 + u_G**2 + u_B**2)

            log_total = np.log(T + eps)
            log_current = np.log(current_uA + eps)
            log_ambient = np.log(amb_mean + eps)

            u_log_total = u_total / (T + eps)

            R_norm = R / (T + eps)
            G_norm = G / (T + eps)
            B_norm = B / (T + eps)

            u_R_norm = R_norm * np.sqrt((u_R / (R + eps))**2 + (u_total / (T + eps))**2)
            u_G_norm = G_norm * np.sqrt((u_G / (G + eps))**2 + (u_total / (T + eps))**2)
            u_B_norm = B_norm * np.sqrt((u_B / (B + eps))**2 + (u_total / (T + eps))**2)

            R_over_G = R / (G + eps)
            R_over_B = R / (B + eps)
            G_over_B = G / (B + eps)

            u_R_over_G = R_over_G * np.sqrt((u_R / (R + eps))**2 + (u_G / (G + eps))**2)
            u_R_over_B = R_over_B * np.sqrt((u_R / (R + eps))**2 + (u_B / (B + eps))**2)
            u_G_over_B = G_over_B * np.sqrt((u_G / (G + eps))**2 + (u_B / (B + eps))**2)

            log_R_over_G = np.log(R_over_G + eps)
            log_R_over_B = np.log(R_over_B + eps)
            log_G_over_B = np.log(G_over_B + eps)

            u_log_R_over_G = np.sqrt((u_R / (R + eps))**2 + (u_G / (G + eps))**2)
            u_log_R_over_B = np.sqrt((u_R / (R + eps))**2 + (u_B / (B + eps))**2)
            u_log_G_over_B = np.sqrt((u_G / (G + eps))**2 + (u_B / (B + eps))**2)

            u_log_ambient = u_amb / (amb_mean + eps)

            # -----------------------------
            # Classifier features (24 features)
            # -----------------------------
            # features_cls = pd.DataFrame([{
            #     "log_total": log_total,
            #     "log_current": log_current,
            #     # "log_ambient": log_ambient,
            #     "R_norm": R_norm,
            #     "G_norm": G_norm,
            #     "B_norm": B_norm,
            #     "R_over_G": R_over_G,
            #     "R_over_B": R_over_B,
            #     "G_over_B": G_over_B,
            #     "log_R_over_G": log_R_over_G,
            #     "log_R_over_B": log_R_over_B,
            #     "log_G_over_B": log_G_over_B,
            #     "u_total": u_total,
            #     "u_log_total": u_log_total,
            #     # "u_log_ambient": u_log_ambient,
            #     "u_R_norm": u_R_norm,
            #     "u_G_norm": u_G_norm,
            #     "u_B_norm": u_B_norm,
            #     "u_R_over_G": u_R_over_G,
            #     "u_R_over_B": u_R_over_B,
            #     "u_G_over_B": u_G_over_B,
            #     "u_log_R_over_G": u_log_R_over_G,
            #     "u_log_R_over_B": u_log_R_over_B,
            #     "u_log_G_over_B": u_log_G_over_B
            # }])

            features_cls = pd.DataFrame([{
                "log_total": log_total,
                "log_current": log_current,
                # "log_ambient",
                "R_norm": R_norm,
                "G_norm": G_norm,
                "B_norm": B_norm,
                "R_over_G": R_over_G,
                "R_over_B": R_over_B,
                "G_over_B": G_over_B,
                "log_R_over_G": log_R_over_G,
                "log_R_over_B": log_R_over_B,
                "log_G_over_B": log_G_over_B
            }])


            features_cls = features_cls[cls_feature_columns]

            # -----------------------------
            # Color prediction
            # -----------------------------
            probs = model_cls.predict_proba(features_cls)[0]
            best_idx = int(np.argmax(probs))
            predicted_label = label_encoder.inverse_transform([best_idx])[0]

            # -----------------------------
            # Distance regression features
            # -----------------------------
            features_dist = build_distance_features(
                log_total=log_total,
                log_current=log_current,
                log_ambient=log_ambient,
                R_norm=R_norm,
                G_norm=G_norm,
                B_norm=B_norm,
                R_over_G=R_over_G,
                R_over_B=R_over_B,
                G_over_B=G_over_B,
                log_R_over_G=log_R_over_G,
                log_R_over_B=log_R_over_B,
                log_G_over_B=log_G_over_B,
                color_label=predicted_label,
                distance_feature_columns=distance_feature_columns
            )

            pred_log_distance = model_reg.predict(features_dist)[0]
            pred_distance_mm = float(np.exp(pred_log_distance))

            # -----------------------------
            # Output
            # -----------------------------
            print(f"\nAmbient mean: {amb_mean:.3f} ± {u_amb:.3f}")
            print(f"R mean      : {R:.3f} ± {u_R:.3f}")
            print(f"G mean      : {G:.3f} ± {u_G:.3f}")
            print(f"B mean      : {B:.3f} ± {u_B:.3f}")
            print(f"Total       : {T:.3f} ± {u_total:.3f}")

            print(f"\nPredicted color: {predicted_label}")
            print("Probabilities:")
            for cls, p in zip(label_encoder.classes_, probs):
                print(f"  {cls:>8}: {p:.3f}")

            print("")
            print(predicted_label + ": " + str(probs[best_idx]))
            print(f"Predicted distance: {pred_distance_mm:.2f} mm")
            if pred_distance_mm < 20 or pred_distance_mm > 140:
                print("Warning: predicted distance is outside training range (20-140 mm)")

            print("-" * 50)

    finally:
        ser.close()


if __name__ == "__main__":
    main()