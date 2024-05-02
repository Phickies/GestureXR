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
from keras import layers, optimizers, losses, metrics, saving, Sequential

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure reproducibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONHASHSEED'] = '42'


class GestureClassifier:

    def __init__(self, input_shape):
        """
        GestureClassifier
        :param input_shape: shape of input data
        """
        print("GestureClassifier Initializing...")
        self.model = Sequential([
            layers.InputLayer(input_shape=input_shape),
            layers.Dense(32, activation='sigmoid'),
            layers.Dense(32, activation='sigmoid'),
            layers.Dense(5, activation='softmax'),
        ])
        self.model.summary()
        self.history = None

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
        test_loss, test_acc = self.model.evaluate(x_test, verbose=0)
        print('Test loss:', test_loss)
        print('Test accuracy:', test_acc)

        plt.figure(figsize=(10, 8))
        prediction = self.model.predict(x_test)
        prediction_classes = np.argmax(prediction, axis=1)
        cm = confusion_matrix(y_test, prediction_classes)
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


if __name__ == "__main__":
    pass
