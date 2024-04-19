import serial
import csv
from datetime import datetime
import keyboard

# Open serial port for receiving data from Arduino
# ser = serial.Serial('COM6', 921600, timeout=1)
ser = serial.Serial('COM7', 921600, timeout=1)

# Create a CSV file
csv_file_path = 'python1.csv'

# Open file for data logging+
file = open(csv_file_path, "w", newline='')
csv_writer = csv.writer(file)


try:
    # Write CSV header with Timestamp, sensor data, label and sep columns
    csv_header = ["Timestamp", "data",
                  "Label", "Sep"]
    csv_writer.writerow(csv_header)

    # Initialize label and previous sign
    labels = ["pinch2finger", "release2finger", "pinch3finger", "release3finger", "fists"]  # List of possible labels
    label_index = 0  # Index for cycling through labels
    current_label = labels[label_index]
    sep = False  # Initial sep value


    def clean_and_join_numbers(data_str):
        cleaned_numbers = ''.join(data_str.replace('-', ''))
        return cleaned_numbers

    while True:
        # Read data from Arduino
        # for _ in range(3):
        line = ser.readline().decode().strip()
        data = line.strip("'").split("\t\t")
        data_row = [clean_and_join_numbers(data_str) for data_str in data]
        # print(len(data))

        # if len(data) == 1:
        #     data_row.extend(data)
        print(data_row)

        if len(data_row) == 1:  # Ensure we have exactly 6 sensor values
            # Get current timestamp1
            current_timestamp = datetime.now().strftime('%H:%M:%S.%f')

            # Check keyboard input to set sep and current_label
            if keyboard.is_pressed('1'):
                current_label = labels[0]
                sep = True
            elif keyboard.is_pressed('2'):
                current_label = labels[1]
                sep = True
            elif keyboard.is_pressed('3'):
                current_label = labels[2]
                sep = True
            elif keyboard.is_pressed('4'):
                current_label = labels[3]
                sep = True
            elif keyboard.is_pressed('5'):
                current_label = labels[4]
                sep = True
            else:
                sep = False

            # Write data to CSV file along with label and sep
            # print(data_row)
            csv_writer.writerow([current_timestamp] + data_row + [current_label, sep])


except serial.SerialException as e:
    print(f"Serial communication error: {e}")

finally:
    # Close serial port
    ser.close()

    # Close file and save data to CSV
    file.close()
    print("Data saved to", csv_file_path)
