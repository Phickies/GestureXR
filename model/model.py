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

from sklearn.metrics import confusion_matrix
from keras import models, layers, optimizers, losses, metrics, saving, Sequential

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure reproducibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONHASHSEED'] = '42'


class GestureClassifier:

    def __init__(self):
        self.model = None
        self.history = None

    def init(self,  input_shape):
        print("GestureClassifier Initializing...")
        self.model = Sequential([
            layers.InputLayer(input_shape=input_shape),
            layers.Dense(32, activation='sigmoid'),
            layers.Dense(32, activation='sigmoid'),
            layers.Flatten(),
            layers.Dense(32, activation='sigmoid'),
            layers.Dense(5, activation='softmax'),
        ])
        self.model.summary()

    def compile(self, learning_rate):
        """
        Compiles the model and sets up optimizer and loss function
        :param learning_rate: learning rate
        :return: none
        """
        print('Compiling the model...')
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=losses.categorical_crossentropy,
            metrics=['accuracy']
        )

    def train(self, X_train, y_train, X_val, y_val, n_epoch: int, batch_size: int):
        """
        Trains the model on the training set.
        :param X_train: numpy array training set
        :param y_train: numpy array label set
        :param X_val: numpy array validation set
        :param y_val: numpy array label set
        :param n_epoch: number of epochs
        :param batch_size: batch size
        :return: history_training
        """
        print('Training the model...')
        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=n_epoch,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
        return self.history

    def plot_history(self):
        """
        Plot the training progress
        :rtype: None
        """
        print('Plotting the training history...')
        if self.history is None:
            raise Exception("No training history found. Train the model first.")

        plt.figure(figsize=(10, 8))
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.show()

    def show_evaluation(self, x_test: np.array, y_test: np.array):
        """
        plot_confusion matrix
        :param x_test: x_test
        :param y_test: y_true
        :rtype: None
        """
        print('Evaluating the model...')
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', test_loss)
        print('Test accuracy:', test_acc)

        plt.figure(figsize=(10, 8))
        prediction = self.model.predict(x_test)
        prediction_classes = np.argmax(prediction, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        cm = confusion_matrix(true_classes, prediction_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='pink')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.show()

    def save_model(self, folder_path_name='gestureXR_model'):
        """
        Save model
        :param folder_path_name: path to save_file
        :rtype: None
        """
        print('Saving model...')
        folder_path_name = folder_path_name + ".keras"
        saving.save_model(self.model, folder_path_name, overwrite=True)
        print(f"Model saved as: {folder_path_name}")

    def load_model(self, zip_path_name):
        """
        Load model
        :param zip_path_name: path to save_file
        :rtype: None
        """
        self.model = saving.load_model(zip_path_name)
        print("loaded model")


class LSTMClassifier:
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

    def save_model(self, folder_path_name='lstm_model'):
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
