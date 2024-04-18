"""
Testing area for Tuning HYPER-PARAMETER
"""

# Import module
import os

import pandas as pd

from model import QuartzClassifier
from helper import convert_tflite_to_c

# Import scikit-learn
from sklearn.model_selection import train_test_split

import numpy as np

data = pd.read_csv("../data/data_all_frame_csv/data.csv",)

df = data.copy()
df2 = data.copy()
# df = df.apply(pd.to_numeric, errors='ignore')
# df2 = df2.apply(pd.to_numeric, errors='ignore')
# df2.dropna(inplace=True)
# df.dropna(inplace=True)




X = df.drop(['Timestamp','Sep','Label_fists','Label_pinch2finger','Label_pinch3finger','Label_release2finger','Label_release3finger'], axis = 1)
X = np.array(X)
#X = np.float32(X)

print(X)
y = df2.drop(['Timestamp','Accel','Gyr','Sep','Label_fists'], axis = 1)
y = np.array(y)
y = np.float32(y)
print(y)

# X = np.random.random((1000, 20, 1))
# y = np.random.randint(2, size=(1000, 3))

# def drop_remain(X,y):
#     remain = X.__len__() % 16
#     if remain == 0:
#         return X,y
#     else:
#         X = X[:-remain, :, :]
#         y = y[:-remain]
#         return X,y


# Create test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
print(X_train.shape, X_val.shape, X_test.shape)

# X_train, y_train = drop_remain(X_train, y_train)
# X_test, y_test = drop_remain(X_test, y_test)
# X_val, y_val = drop_remain(X_val, y_val)

# Create an instance of the classifier

model = QuartzClassifier(output_unit=4, n_batchs=16)

# Initialize and train the model (assuming X and y are already defined and preprocessed)
model.initialize(input_shape=X_test.shape[1:])
model.train(X_train, X_val, y_train, y_val)

# Evaluate the model
model.evaluate(X_test, y_test)


# Plot training history
model.plot_history()

# Plot confusion matrix
model.plot_confusion_matrix(y_test)

# Save the model
model.save_model()
