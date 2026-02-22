import serial
import time
import sys

# Open and parse drawn digit from text file (28 lines of 28 space-separated pixel values)
def load_digit(filename="drawn_digit.txt"):
    pixels = []
    with open(filename, 'r') as f:
        for line in f:
            for num in line.strip().split():
                pixels.append(int(num))
    return pixels[:784]

if __name__ == "__main__":
    # Change to what port your Arduino is on  (ex: 'COM3' on Windows or '/dev/ttyACM0' on Linux)
    port = 'COM6'
    
    pixels = load_digit()
    print(f"Sending {len(pixels)} pixels to {port}...")

    try:
        ser = serial.Serial()
        ser.port = port
        ser.baudrate = 115200
        ser.dtr = False
        ser.rts = False
        ser.open()
        time.sleep(2.5)

        ser.write(bytes(pixels))
        ser.flush()
        print("Sent.")

        # Wait up to 5 seconds for the prediction
        start = time.time()
        while time.time() - start < 5.0:
            if ser.in_waiting > 0:
                resp = ser.readline().decode(errors='ignore').strip()
                if resp.startswith("PREDICTION:"):
                    print("Arduino result:", resp.split(":", 1)[1])
                    break
            time.sleep(0.05)

        ser.close()
    except Exception as e:
        print("Error:", e)