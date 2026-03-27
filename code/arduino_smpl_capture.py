import struct
import time
import serial
import matplotlib.pyplot as plt


PORT = "COM7"
BAUD = 115200
START_CMD = b"r"
END_CHAR = b"\n"

def read_exact(ser: serial.Serial, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = ser.read(size - len(data))
        if not chunk:
            raise TimeoutError(f"Timeout while reading {size} bytes")
        data.extend(chunk)
    return bytes(data)

def request_samples(ser: serial.Serial):
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    ser.write(START_CMD)
    ser.flush()

    header = read_exact(ser, 2)
    sample_count = struct.unpack("<H", header)[0]

    payload = read_exact(ser, sample_count * 2)
    samples = list(struct.unpack(f"<{sample_count}h", payload))

    footer = read_exact(ser, 1)
    if footer != END_CHAR:
        raise ValueError(f"Bad end character: {footer!r}")

    return sample_count, samples

def main():
    with serial.Serial(PORT, BAUD, timeout=2) as ser:
        time.sleep(2.0)   # important for Uno reset
        ser.reset_input_buffer()

        count, samples = request_samples(ser)
        # print(f"Received {count} samples")
        # print(samples[:10])
        plt.figure()
        # plt.stem(samples)
        plt.plot(samples)
        plt.show()


if __name__ == "__main__":
    main()