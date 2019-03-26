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


class NeuralNetwork:
    def __init__(self, csv_file_path,
                 csv_file_test_data_size_in_percents=75,
                 is_shuffled_data_required=True,
                 training_iterations=10,
                 learning_rate=0.1):
        # read file
        csv_data = read_csv(csv_file_path, delimiter=',')
        # shuffle file data
        if is_shuffled_data_required:
            csv_data = self.shuffle_data(csv_data)
        test_size = csv_file_test_data_size_in_percents/100
        # file data headers
        csv_data_headers = [header for header in csv_data]
        self.features = csv_data_headers[:-1]
        self.label = csv_data_headers[-1]
        # training_iterations
        x_data = csv_data.drop(
            ["Classification"], axis=1).values
        # read file data values
        y_data = csv_data["Classification"].values
        # split data values into training and test sets
        x_data_training, x_data_testing, y_data_training, y_data_testing = train_test_split(
            x_data, y_data, test_size=test_size, random_state=0)
        # normalize data values
        sc = StandardScaler()
        sc.fit(x_data_training)
        x_data_training = sc.fit_transform(x_data_training)
        x_data_testing = sc.transform(x_data_testing)
        # self.simulate_training(100, 0.1, x_data, y_data)
        # test deep_belief_network
        self.deep_belief_network_prediction(
            x_data_training, x_data_testing, y_data_training, y_data_testing, learning_rate, training_iterations)
        # self.convolutional_neural_network_prediction(x_data_training, x_data_testing, y_data_training, y_data_testing, training_iterations)
        # self.perceptron_network_prediction(x_data_training, x_data_testing, y_data_training, y_data_testing, training_iterations)
        # classifier = tf.estimator.DNNClassifier(
        #     feature_columns=[tf.feature_column.numeric_column(key) for key in self.features],
        #     hidden_units=hidden_units_spec,
        #     n_classes=n_classes_spec,
        #     model_dir=tmp_dir_spec)

    @staticmethod
    def shuffle_data(data):
        return data.reindex(np.random.permutation(data.index))

    @staticmethod
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    @staticmethod
    def derivatives_sigmoid(x):
        return x * (1 - x)

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
            classifier.save("dbn_model.pk1")
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
            mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=training_iterations)  
            mlp.fit(x_data_training, y_data_training)
            predictions = mlp.predict(x_data_testing) 
            classifier_accuracy = accuracy_score(y_data_testing, predictions)
            print(classifier_accuracy)
    
    def deep_neural_network_prediction(self, x_data_training, x_data_testing,  y_data_training, y_data_testing, training_iterations):
        classifier = tf.estimator.DNNClassifier(
            feature_columns=self.features, 
            hidden_units=[9,18,9], 
            n_classes=2, 
            model_dir='model')
        
        train_input_fn = tf.estimator.inputs.pandas_input_fn(
                    x=x_data_training, 
                    y=y_data_training, 
                    num_epochs=training_iterations, 
                    shuffle=True)

        classifier.train(input_fn=train_input_fn, steps=100)

        test_input_fn = tf.estimator.inputs.pandas_input_fn(
                    x=x_data_testing, 
                    y=y_data_testing, 
                    num_epochs=training_iterations, 
                    shuffle=False)
        
        predict_input_fn = tf.estimator.inputs.pandas_input_fn(
                      x=x_data_testing, 
                      num_epochs=1, 
                      shuffle=False)

        predictions = list(classifier.predict(input_fn=predict_input_fn))

    def simulate_training(self, training_iterations, learning_rate, x, y):
        number_of_input_layer_neurons = len(self.features)
        number_of_hidden_layer_neurons = 3
        number_of_output_neurons = len(self.label)

        weights = np.random.uniform(
            size=(number_of_input_layer_neurons, number_of_hidden_layer_neurons))
        bias = np.random.uniform(size=(1, number_of_hidden_layer_neurons))
        weights_out = np.random.uniform(
            size=(number_of_hidden_layer_neurons, number_of_output_neurons))
        bias_out = np.random.uniform(size=(1, number_of_output_neurons))
        for i in range(training_iterations):
            # Forward Propogation
            hidden_layer_input1 = np.dot(x, weights)
            hidden_layer_input = hidden_layer_input1 + bias
            hiddenlayer_activations = self.sigmoid(hidden_layer_input)
            output_layer_input1 = np.dot(hiddenlayer_activations, weights_out)
            output_layer_input = output_layer_input1 + bias_out
            output = self.sigmoid(output_layer_input)
            # Backpropagation
            E = y-output
            slope_output_layer = self.derivatives_sigmoid(output)
            slope_hidden_layer = self.derivatives_sigmoid(
                hiddenlayer_activations)
            d_output = E * slope_output_layer
            Error_at_hidden_layer = d_output.dot(weights_out.T)
            d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
            weights_out += hiddenlayer_activations.T.dot(
                d_output) * learning_rate
            bias_out += np.sum(d_output, axis=0, keepdims=True) * learning_rate
            weights += x.T.dot(d_hiddenlayer) * learning_rate
            bias += np.sum(d_hiddenlayer, axis=0,
                           keepdims=True) * learning_rate
        print(output)


nn = NeuralNetwork(csv_file_path="dataR2.csv",
                   training_iterations=10, learning_rate=0.01,
                   csv_file_test_data_size_in_percents=20)
