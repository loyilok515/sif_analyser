import serial

ser = serial.Serial('/dev/ttyUSB0', 57600, timeout=1)
print("Receiver ready...")

def checksum(data: str) -> str:
    result = 0
    for c in data:
        result ^= ord(c)
    return f"{result:02X}"

while True:
    line = ser.readline()
    if not line:
        continue
    try:
        decoded = line.decode().strip()

        # Split payload and checksum on '*'
        payload, received_cs = decoded.split('*')
        expected_cs = checksum(payload)

        if received_cs != expected_cs:
            print(f"Checksum FAIL — expected {expected_cs}, got {received_cs}  raw: {decoded}")
            continue

        lat, lon = payload.split(',')
        print(f"OK  Lat: {float(lat):.6f}  Lon: {float(lon):.6f}")

    except ValueError:
        print(f"Bad packet: {line}")