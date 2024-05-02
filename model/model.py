"""
Module containing a QuartzClassifier class that can
initialize, train, evaluate, and save a deep LSTM model using Keras.

** WARNING **
This file only use for CUSTOM MODEL, adding layer, changing model behaviour, adjusting tensors
Tuning HYPER-PARAMETERS should be in test.py file
** WARNING **

"""
# Import system
import os
import math
import numpy as np
import mouse
import serial
import csv
from datetime import datetime
import keyboard

# Ensure reproducibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONHASHSEED'] = '42'

# Import scikit-learn and its dependencies
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

# import tensorflow and its dependencies
from keras import models, layers, optimizers, losses, metrics, saving, callbacks

# Import data visual library
import seaborn as sns
import matplotlib.pyplot as plt


class QuartzClassifier:

    def __init__(self, output_unit=3, drop_out_rate=0.6,
                 learning_rate=0.01, n_epochs=20, n_batchs=128, prediction_threshold=0.5, model=None):
        """
        :type prediction_threshold: object
        :param output_unit: number of classification output
        :param drop_out_rate: drop out rate from the last tensor
        :param learning_rate: learning rate
        :param n_epochs: number of epoch
        :param n_batchs: number of batch
        :param prediction_threshold: Don't need to change
        """
        self.output_unit = output_unit
        self.drop_out_rate = drop_out_rate
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_batchs = n_batchs
        self.prediction_threshold = prediction_threshold
        self.model = model
        self.history = None
        self.cp_callback = None
        self.y_pred = 0
        self.gesture = 0

    def initialize(self, input_shape):
        """
        Initialize the LSTM model
        :rtype: None
        :param input_shape: numpy array shape
        """
        self.model = models.Sequential([
            layers.Input(shape=input_shape, batch_size=self.n_batchs),
            layers.LSTM(units=128, stateful=True, return_sequences=True),
            layers.LSTM(units=128, stateful=True, return_sequences=True),
            layers.LSTM(units=64, stateful=True, return_sequences=False),
            layers.Dense(units=32,activation='sigmoid'),
            layers.Dropout(self.drop_out_rate),
            layers.Dense(self.output_unit, activation='softmax')
        ])

        self.model.compile(optimizer='adam',
                           loss=losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])

        self.model.summary()

    def set_checkpoint(self, checkpoint_path: str, n_sample: int, saving_rate: int = 5):
        """
        Create a check point for train and saving weight for further investigate performance
        See checkpoint callback usage from tensorflow document for more detail
        :param n_sample: Number of sample for training
        :param saving_rate: frequency of saving weight per batch
        :param checkpoint_path: path for saving weight.
        """
        n_batches = math.ceil(n_sample / self.n_batchs)

        # Create a callback that saves the model's weights, put into list
        self.cp_callback = [callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                      save_weights_only=True,
                                                      save_freq=saving_rate * n_batches,
                                                      verbose=1)]

    def train(self, X_train, X_val, y_train, y_val, verbose=2):
        """
        Train the model
        :param X_train: training set
        :param X_val: validation set
        :param y_train: label training set
        :param y_val: label validation set
        :param verbose: showing progress
        :rtype: None
        """

        if self.model is None:
            raise Exception("Model has not been initialized. Call initialize_model() first.")

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.n_epochs,
            batch_size=self.n_batchs,
            shuffle=False,
            verbose=verbose,
            callbacks=self.cp_callback

        )


    def collect(self):
        # Open serial port for receiving data from Arduino
        # ser = serial.Serial('COM6', 921600, timeout=1)
        ser = serial.Serial('COM5', 921600, timeout=1)

        # Create a CSV file
        csv_file_path = 'data/data290.csv'

        # Open file for data logging+
        file = open(csv_file_path, "w", newline='')
        csv_writer = csv.writer(file)

        try:
            # Write CSV header with Timestamp, sensor data, label and sep columns
            csv_header = ["Timestamp", "Accel", "Gyr", "Label", "Sep"]
            csv_writer.writerow(csv_header)

            # Initialize label and previous sign
            labels = ["fists", "release2finger", "pinch3finger", "release3finger"]  # List of possible labels
            label_index = 0  # Index for cycling through labels
            current_label = labels[label_index]
            sep = False  # Initial sep value

            def clean_and_join_numbers(data_str):
                # 将字符串中的连字符 '-' 替换为空格，然后使用空格连接所有数字
                cleaned_numbers = ''.join(data_str.replace('-', ''))
                return cleaned_numbers
            count = 1
            while count == 1:
                data_row = []
                # Read data from Arduino
                # for _ in range(3):
                line = ser.readline().decode().strip()
                # print(line)
                data = line.strip("'").split("\t\t")
                data = line.split(",")
                # print(data)
                # data_row = [clean_and_join_numbers(data_str) for data_str in line]
                # print(data_row)
                numbers = [num for num in data]

                # 将数字列表按照每个子列表包含9个数字的方式进行重组
                result = [numbers[i:i + 9] for i in range(0, len(numbers), 9)]
                print(result)
                print(len(result))

                # if len(data) == 1:
                #     data_row.extend(data)
                # print(data_row)

                # if len(result) == 2:  # Ensure we have exactly 6 sensor values
                # Get current timestamp1
                current_timestamp = datetime.now().strftime('%H:%M:%S.%f')
                # Check keyboard input to set sep and current_label
                if keyboard.is_pressed('4'):
                    current_label = labels[3]
                    sep = False
                    count = 0
                    break
                # elif keyboard.is_pressed('5'):
                #     current_label = labels[4]
                #     sep = True
                else:
                    sep = False

                # Write data to CSV file along with label and sep
                # print(data_row)
                # print(data_row)
                csv_writer.writerow([current_timestamp] + result + [current_label, sep])



        except serial.SerialException as e:
            print(f"Serial communication error: {e}")

        finally:
            # Close serial port
            ser.close()

            # Close file and save data to CSV
            file.close()
            print("Data saved to", csv_file_path)

    def mouse(self):

        if self.gesture == 0:
            mouse.wheel(10)
        elif self.gesture == 1:
            mouse.wheel(-10)
        elif self.gesture == 2:
            mouse.click('left')
        elif self.gesture == 3:
            mouse.click('right')
        elif self.gesture == 4:
            mouse.move(150, 150)

    def predict(self,X):
        y_pred = self.model.predict(X, batch_size=self.n_batchs, verbose=2)
        print(y_pred)
        y_pred = np.argmax(y_pred, axis=1)
        self.y_pred = y_pred
        self.gesture = self.model.predict(np.expand_dims(X[0], axis=0))[0].argmax()
        print(y_pred)
        print(self.gesture)
        return self.gesture



    def evaluate(self, X_test, y_test):
        """
        Evaluate model, print test lost, test accuracy
        :param X_test: training test set
        :param y_test: label test set
        :rtype: tuple
        :return: accuracy score, recall score, precision score
        """
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, batch_size=self.n_batchs, verbose=2)

        print(f"Test lost: {test_loss}")
        print(f"Test accuracy: {test_accuracy}")

        y_pred = self.model.predict(X_test, batch_size=self.n_batchs, verbose=2)
        print(y_pred)
        y_pred = np.argmax(y_pred, axis=1)
        self.y_pred = y_pred
        self.gesture = self.model.predict(np.expand_dims(X_test[0], axis=0))[0].argmax()
        print(y_pred)
        print(self.gesture)

        # accuracy = accuracy_score(y_test, y_pred)
        # recall = recall_score(y_test, y_pred)
        # precision = precision_score(y_test, y_pred)

        return

    def plot_history(self):
        """
        Plot the training progress
        :rtype: None
        """
        if self.history is None:
            raise Exception("No training history found. Train the model first.")

        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.show()

    def plot_confusion_matrix(self, y_test):
        """
        plot_confusion matrix
        :param y_test: y_true
        :param y_pred: y_prediction
        :rtype: None
        """
        y_test = np.argmax(y_test, axis=1)
        cm = confusion_matrix(y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='pink', xticklabels=range(self.output_unit),
                    yticklabels=range(self.output_unit))
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.show()

    def save_model(self, folder_path_name='quartz_model'):
        """
        Save model
        :param folder_path_name: path to save_file
        :rtype: None
        """
        folder_path_name = folder_path_name + ".keras"
        saving.save_model(self.model, folder_path_name, overwrite=True)
        print(f"Model saved as: {folder_path_name}")

    def load_model(self, zip_path_name):

        self.model = saving.load_model(zip_path_name)
        print("loaded model")


if __name__ == "__main__":
    pass
