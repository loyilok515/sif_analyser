import serial
import time

ser = serial.Serial('/dev/ttyUSB0', 57600, timeout=1)
print("Sender ready...")

lat =  42.123456
lon = -80.654321

def checksum(data: str) -> str:
    result = 0
    for c in data:
        result ^= ord(c)
    return f"{result:02X}"

while True:
    payload = f"{lat},{lon}"
    msg     = f"{payload}*{checksum(payload)}\n"
    ser.write(msg.encode())
    print(f"Sent: {msg.strip()}")
    time.sleep(1)