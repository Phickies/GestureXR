"""
Testing area for Tuning HYPER-PARAMETER
"""
import numpy as np

from model import GestureClassifier
from data.data_preprocess import get_data
from sklearn.model_selection import train_test_split

X, y = get_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=69)

model = GestureClassifier(X.shape[1:])
model.compile(0.001)
model.train(X_train, y_train, X_val, y_val, 10, batch_size=10)
model.plot_history()
model.show_evaluation(X_test, y_test)
model.save_model()
model.load_model("quartz_model.keras")
