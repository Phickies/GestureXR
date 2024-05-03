import socket
import numpy as np
from model import GestureClassifier

# ESP32's IP address (check the Serial Monitor for the IP address when ESP32 connects to WiFi)
ESP_IP = '192.168.178.78'
ESP_PORT = 80

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((ESP_IP, ESP_PORT))

label = ['fist', 'pinch2finger', 'pinch3finger', 'release2finger', 'release3finger']

# Send a request (optional)
s.sendall(b'Hello ESP32!')
#
model = GestureClassifier()
model.load_model("gestureXR_model.keras")

try:
    while True:
        input_data = []
        data = s.recv(1024)  # Receive data from the server (max 1024byte)
        if not data:
            break

        # Decode the byte string to a regular string
        decoded_data = data.decode('utf-8')
        lines = decoded_data.strip().split("\r\n")
        all_float_numbers = []
        for line in lines:
            try:
                # Split the line by commas
                string_numbers = line.split(',')
                # Convert each string to a float and append to the list
                float_numbers = [float(num) for num in string_numbers]
                all_float_numbers.append(float_numbers)
            except ValueError:
                all_float_numbers.append(-1)
        # Print the result
        for numbers in all_float_numbers:
            print(numbers)
            if not isinstance(numbers, int) and len(numbers) == 12:
                numbers = np.asarray(numbers)
                numbers = numbers.reshape(1, -1)  # Reshape to add a batch dimension
                print(numbers)
                prediction = model.model.predict(numbers, verbose=0)
                print(label[np.argmax(prediction)])
            else:
                print("Reading error")

finally:
    s.close()
    print('Connection closed')