"""
Testing area for Tuning HYPER-PARAMETER
"""
import numpy as np

from model import GestureClassifier, LSTMClassifier
from data.data_preprocess import get_data
from sklearn.model_selection import train_test_split
from helper import drop_remain

X, y = get_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=69)

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

# model = GestureClassifier(X.shape[1:])
# model.compile(0.01)
# model.train(X_train, y_train, X_val, y_val, 5, 10)
# model.plot_history()
# model.show_evaluation(X_test, y_test)
# model.save_model()

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

X_train, y_train = drop_remain(X_train, y_train, 128)
X_val, y_val = drop_remain(X_val, y_val, 128)
X_test, y_test = drop_remain(X_test, y_test, 128)

model = LSTMClassifier(output_unit=5, n_batchs=128)
model.initialize(X_train.shape[1:])
model.train(X_train, X_val, y_train, y_val)
model.evaluate(X_test, y_test)
model.plot_history()
model.plot_confusion_matrix(y_test)
model.save_model()
