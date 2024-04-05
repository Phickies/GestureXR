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

# Ensure reproducibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONHASHSEED'] = '42'

# Import scikit-learn and its dependencies
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

# import tensorflow and its dependencies
from keras import models, layers, optimizers, losses, metrics

# Import data visual library
import seaborn as sns
import matplotlib.pyplot as plt

class QuartzClassifier:

    def __init__(self, output_unit=3, drop_out_rate=0.5,
                 learning_rate=0.001, n_epochs=10, n_batchs=32, prediction_threshold=0.5, model=None):
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

    def initialize(self, input_shape):
        """
        Initialize the LSTM model
        :rtype: None
        :param input_shape: numpy array shape
        """
        self.model = models.Sequential([
            layers.Input(shape=input_shape, batch_size=self.n_batchs),
            layers.LSTM(units=128, stateful=True, return_sequences=True),
            layers.LSTM(units=64, stateful=True, return_sequences=True),
            layers.LSTM(units=64, stateful=True, return_sequences=False),
            layers.Dense(units=32, activation='relu'),
            layers.Dropout(self.drop_out_rate),
            layers.Dense(self.output_unit, activation='softmax')
        ])

        self.model.compile(optimizer=optimizers.SGD(learning_rate=self.learning_rate),
                           loss=losses.CategoricalCrossentropy(),
                           metrics=[metrics.CategoricalAccuracy()])

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
            verbose=verbose
        )

    def evaluate(self, X_test, y_test):
        """
        Evaluate model, print test lost, test accuracy
        :param X_test: training test set
        :param y_test: label test set
        :rtype: tuple
        :return: accuracy score, recall score, precision score
        """
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=2)

        print(f"Test lost: {test_loss}")
        print(f"Test accuracy: {test_accuracy}")

        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        return accuracy, recall, precision

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

    def plot_confusion_matrix(self, y_test, y_pred):
        """
        plot_confusion matrix
        :param y_test: y_true
        :param y_pred: y_prediction
        :rtype: None
        """
        cm = confusion_matrix(y_test, y_pred)
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
        self.model.save(folder_path_name)
        print(f"Model saved as: {folder_path_name}")


if __name__ == "__main__":
    pass
