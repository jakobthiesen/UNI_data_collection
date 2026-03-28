import struct
import time
import serial
import matplotlib.pyplot as plt
import numpy as np

PORT = "COM3"
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

        print(f"{i+1:3d}/{n_samples} | Amb = {amb:8.3f}, R={r:8.3f}, G={g:8.3f}, B={b:8.3f}")

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


def main():
    ser = open_serial(PORT)

    try:
        current = 128  # ~1 mA
        print("Capturing 100 samples at ~1 mA...\n")

        amb_vals, r_vals, g_vals, b_vals = capture_series(ser, current, 10)
        print("R: " + str(np.mean(r_vals)))
        print("G: " + str(np.mean(g_vals)))
        print("B: " + str(np.mean(b_vals)))

        # plot_data(amb_vals, r_vals, g_vals, b_vals)

    finally:
        ser.close()


if __name__ == "__main__":
    main()