"""
Testing area for Tuning HYPER-PARAMETER
"""

# Import module
import os
import pandas as pd
import numpy as np

from model import QuartzClassifier
from data import data_preprocess

# Import scikit-learn
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)

df = data_preprocess.get_data()

print(df.head)
print(df.dtypes)

print("Start convert data")
X = df.drop(['Timestamp', 'Sep'], axis=1)
total = []
for index, row in X.iterrows():
    twoD_list = [row['Accel'], row['Gyr']]
    total.append(twoD_list)
X = np.array(total)
print(X)

y = df.drop(['Timestamp', 'Accel', 'Gyr', 'Sep'], axis=1)
y = np.array(y)
print(y)


# Create test set
print("Start splitting data set")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
# X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
# X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
print(X_train.shape, X_val.shape, X_test.shape)


def drop_remain(a, b):
    remain = a.__len__() % 16
    if remain == 0:
        return a, b
    else:
        a = a[:-remain, :, :]
        b = b[:-remain]
        return a, b



X_train, y_train = drop_remain(X_train, y_train)
X_test, y_test = drop_remain(X_test, y_test)
X_val, y_val = drop_remain(X_val, y_val)

# Create an instance of the classifier

model = QuartzClassifier(output_unit=5, n_batchs=16)

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
