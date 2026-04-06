import struct
import time
import serial
import matplotlib.pyplot as plt
import numpy as np
import csv
import os



PORT = "COM6"
BAUDRATE = 115200
TIMEOUT_S = 5.0

END_CHAR = b"\n"
REPLY_LEN = 17  # 3 floats + '\n'



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


def capture_series(ser, current, n_samples=100):
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

        print(f"{i+1:3d}/{n_samples} | I = {current:8.3f}, Amb = {amb:8.3f}, R={r:8.3f}, G={g:8.3f}, B={b:8.3f}")

    return amb_vals, r_vals, g_vals, b_vals


def plot_data(amb_vals, r_vals, g_vals, b_vals):
    x = list(range(len(r_vals)))

    plt.figure(figsize=(10, 5))
    plt.semilogy(x, amb_vals, label="amb", color = 'Steelblue')
    plt.semilogy(x, r_vals, label="R", color = 'red')
    plt.semilogy(x, g_vals, label="G", color = 'green')
    plt.semilogy(x, b_vals, label="B", color = 'blue')

    plt.xlabel("Sample index")
    plt.ylabel("Measured value")
    plt.title("RGB Measurements (100 samples @ ~1 mA)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import csv
import os
import time
import numpy as np

CSV_FILE = "rgb_data.csv"

# ---- Ensure CSV has header ----
def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "session_id",
                "repeat_id",
                "distance_mm",
                "target_id",
                "led_current_uA",
                "sample_index",

                "ambient",
                "R_response",
                "G_response",
                "B_response",

                "total",
            ])

# ---- Append one batch ----
def append_measurement(session_id, repeat_id, distance_mm, target_id, current_uA,
                       amb_vals, r_vals, g_vals, b_vals):

    amb_vals = np.array(amb_vals)
    r_vals = np.array(r_vals)
    g_vals = np.array(g_vals)
    b_vals = np.array(b_vals)

    # --- Total response (for uncertainty + ratios ONLY) ---
    total_response = r_vals + g_vals + b_vals

    # Avoid division by zero
    total_response_safe = np.where(total_response == 0, 1e-9, total_response)

    # --- Ratios (spectral features) ---
    r_ratio = r_vals / total_response_safe
    g_ratio = g_vals / total_response_safe
    b_ratio = b_vals / total_response_safe

    # --- Uncertainty estimate ---
    # Based on total response variation
    u_estimate = np.std(total_response)

    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        batch_timestamp = time.time()
        for i in range(len(r_vals)):
            sample_index = i
            

            amb = amb_vals[i]
            r_resp = r_vals[i]
            g_resp = g_vals[i]
            b_resp = b_vals[i]

            total = total_response[i]

            writer.writerow([
                batch_timestamp,
                session_id,
                repeat_id,
                distance_mm,
                target_id,
                current_uA,
                sample_index,

                amb,

                r_resp,
                g_resp,
                b_resp,

                total          # useful for ML (distance signal)

            ])




def main():
    ser = open_serial(PORT)

    try:
        # current = 128  # ~1 mA
        # print("Capturing 100 samples at ~1 mA...\n")

        # amb_vals, r_vals, g_vals, b_vals = capture_series(ser, current, 10)
        # print("R: " + str(np.mean(r_vals)))
        # print("G: " + str(np.mean(g_vals)))
        # print("B: " + str(np.mean(b_vals)))

        # plot_data(amb_vals, r_vals, g_vals, b_vals)
        # init_csv()

        n_samples = 25

        session_id = 3
        distance_mm = 75
        target_id = "orange"

        # currents = [500,1000,2000,4000, 8000]
        currents = [4000, 8000]
        tx_currents = [0]*len(currents)
        for n in range(len(tx_currents)):
            tx_currents[n] = int(currents[n]/39.2157)
        # print(tx_currents)

        for i in range(15):
            for n in range(len(currents)):
                current = currents[n]
                tx_current = tx_currents[n]
                amb_vals, r_vals, g_vals, b_vals = capture_series(ser, tx_current, n_samples)
                repeat_id = i+1
                append_measurement(
                    session_id,
                    repeat_id,
                    distance_mm,
                    target_id,
                    current,
                    amb_vals,
                    r_vals,
                    g_vals,
                    b_vals
                )
            time.sleep(0.25)



    finally:
        ser.close()


if __name__ == "__main__":
    main()