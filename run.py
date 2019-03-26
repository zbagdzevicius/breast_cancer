import numpy as np
from pandas import read_csv
from sklearn.metrics.classification import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from dbn import SupervisedDBNClassification
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from csv import writer


class NeuralNetwork:
    def __init__(self, csv_file_path,
                 csv_file_test_data_size_in_percents=75,
                 training_iterations=10,
                 learning_rate=0.1):
        self.test_size = csv_file_test_data_size_in_percents/100
        
        csv_data = read_csv(csv_file_path, delimiter=',')
        self.features, self.label = self.__get_csv_data_headers(csv_data)
        x_data_training, x_data_testing, y_data_training, y_data_testing = self.prepare_training_data(csv_data)
        self.deep_belief_network_prediction(
            x_data_training, x_data_testing, y_data_training, y_data_testing, learning_rate, training_iterations)
        self.convolutional_neural_network_prediction(x_data_training, x_data_testing, y_data_training, y_data_testing, training_iterations)
        self.perceptron_network_prediction(x_data_training, x_data_testing, y_data_training, y_data_testing, training_iterations)

    @staticmethod
    def __shuffle_data(csv_data):
        return csv_data.reindex(np.random.permutation(csv_data.index))
    
    def prepare_training_data(self, csv_data):
        csv_data = self.__shuffle_data(csv_data)
        # input data
        x_data = csv_data.drop(
            ["Classification"], axis=1).values
        # output data
        y_data = csv_data["Classification"].values
        # split data values into training and test sets
        x_data_training, x_data_testing, y_data_training, y_data_testing = train_test_split(
            x_data, y_data, test_size=self.test_size, random_state=0)
        # normalize data values
        x_data_training, x_data_testing = self.__normalize_values(x_data_training, x_data_testing)
        return x_data_training, x_data_testing, y_data_training, y_data_testing
    
    @staticmethod
    def __get_csv_data_headers(csv_data):
        csv_data_headers = [header for header in csv_data]
        features = csv_data_headers[:-1]
        label = csv_data_headers[-1]
        return features, label
    
    @staticmethod
    def __normalize_values(x_data_training, x_data_testing):
        sc = StandardScaler()
        sc.fit(x_data_training)
        x_data_training = sc.fit_transform(x_data_training)
        x_data_testing = sc.transform(x_data_testing)
        return x_data_training, x_data_testing

    def deep_belief_network_prediction(self, x_data_training, x_data_testing,  y_data_training, y_data_testing, learning_rate, training_iterations):
        classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                                learning_rate_rbm=learning_rate/2,
                                                learning_rate=learning_rate,
                                                n_epochs_rbm=int(training_iterations/10),
                                                n_iter_backprop=training_iterations,
                                                batch_size=32,
                                                activation_function='relu',
                                                dropout_p=0.2)
        classifier.fit(x_data_training, y_data_training)
        y_data_prediction = classifier.predict(x_data_testing)
        classifier_accuracy = accuracy_score(y_data_testing, y_data_prediction)
        if classifier_accuracy >= 0.85:
            classifier.save("models/dbn/model.pk1")
        print(classifier_accuracy)

    def convolutional_neural_network_prediction(self, x_data_training, x_data_testing,  y_data_training, y_data_testing, training_iterations):
        # input layer + first hidden layer
        classifier = Sequential()
        classifier.add(Dense(output_dim=len(self.features), init='uniform',
                            activation='relu', input_dim=len(self.features)))
        # second hidden layer
        classifier.add(Dense(output_dim=len(self.features)*4, init='uniform', activation='relu'))
        # output layer
        classifier.add(
            Dense(output_dim=1, init='uniform', activation='sigmoid'))
        classifier.compile(
            optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        classifier.fit(x_data_training, y_data_training,
                    batch_size=32, nb_epoch=training_iterations)
        y_data_prediction = classifier.predict(x_data_testing)
        y_data_prediction = (y_data_prediction > 0.5)
        classifier_accuracy = accuracy_score(y_data_testing, y_data_prediction)
        if classifier_accuracy >= 0.85:
            classifier.save("keras_nn.pk1")
        print(classifier_accuracy)
    
    def perceptron_network_prediction(self, x_data_training, x_data_testing,  y_data_training, y_data_testing, training_iterations):
        for x in range(30):
            mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=training_iterations*13)  
            mlp.fit(x_data_training, y_data_training)
            predictions = mlp.predict(x_data_testing) 
            classifier_accuracy = accuracy_score(y_data_testing, predictions)
            print(classifier_accuracy)

nn = NeuralNetwork(csv_file_path="dataR2.csv",
                   training_iterations=100, learning_rate=0.01,
                   csv_file_test_data_size_in_percents=20)
