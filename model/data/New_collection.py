import serial
import json
from datetime import datetime
import keyboard


# Open serial port for receiving data from Arduino
# ser = serial.Serial('COM6', 921600, timeout=1)
ser = serial.Serial('COM5', 921600, timeout=1)

# Create a CSV file


# Open file for data logging+


labels = ["pinch2finger", "release2finger", "pinch3finger", "release3finger", "fists"]  # List of possible labels
label_index = 0  # Index for cycling through labels
current_label = labels[label_index]
timestamp = []
position = []
label = []
count = 1
while count == 1:
    # Read data from Arduino
    # for _ in range(3):
    line = ser.readline().decode().strip()
    final = {"timestamp": [], "position": [], "label": []}
    data = line.strip("'").split("\t\t")

    print(data)

    # if len(data) == 1:  # Ensure we have exactly 6 sensor values
    #     # Get current timestamp1
    current_timestamp = datetime.now().strftime('%H:%M:%S.%f')

    # Check keyboard input to set sep and current_label
    if keyboard.is_pressed('1'):
        current_label = labels[0]

    elif keyboard.is_pressed('2'):
        current_label = labels[1]

    elif keyboard.is_pressed('3'):
        current_label = labels[2]

    elif keyboard.is_pressed('4'):
        current_label = labels[3]

    elif keyboard.is_pressed('5'):
        current_label = labels[4]

    elif keyboard.is_pressed('6'):
        final = {"timestamp": timestamp, "position": position, "label": label}
        with open("sample41.json", "w") as outfile:
            json.dump(final, outfile)

        ser.close()

    timestamp.append(current_timestamp)
    position.append(data)
    label.append(current_label)


