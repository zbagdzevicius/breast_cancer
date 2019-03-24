import tensorflow as tf
import pandas as pd
import numpy as np

training_set_size_portion = .8
# Keep track of the accuracy score
accuracy_score = 0
# The DNN has hidden units, set the spec for them here
hidden_units_spec = [10,20,10]
n_classes_spec = 2
# The number of training steps
steps_spec = 2000
# The number of epochs
epochs_spec = 15
# Here's a set of our features. If you look at the CSV, 
# you'll see these are the names of the columns. 
# In this case, we'll just use all of them:
features = ['radius','texture']
# Here's the label that we want to predict -- it's also a column in # the CSV
labels = ['diagnosis_numeric']


randomized_data = my_data

total_records = len(randomized_data)
training_set_size = int(total_records * training_set_size_portion)
test_set_size = total_records = training_set_size

# Build the training features and labels
training_features = randomized_data.head(training_set_size)[features].copy()
training_labels = randomized_data.head(training_set_size)[labels].copy()
print(training_features.head())
print(training_labels.head())

# Build the testing features and labels
testing_features = randomized_data.tail(test_set_size)[features].copy()
testing_labels = randomized_data.tail(test_set_size)[labels].copy()

feature_columns = 
    [tf.feature_column.numeric_column(key) for key in features]

classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns, 
    hidden_units=hidden_units_spec, 
    n_classes=n_classes_spec, 
    model_dir=tmp_dir_spec)