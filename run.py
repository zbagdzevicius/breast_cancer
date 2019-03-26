from sklearn.metrics.classification import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from dbn import SupervisedDBNClassification
import numpy as np
from pandas import read_csv
from csv import writer


class NeuralNetwork:
    def __init__(self, csv_file_path, csv_file_test_data_size_in_percents=75):
        self.test_size = csv_file_test_data_size_in_percents / 100
        self.csv_data = read_csv(csv_file_path, delimiter=",")
        self.features, self.label = self.__get_csv_data_headers(self.csv_data)
        self.x_data_training, self.x_data_testing, self.y_data_training, self.y_data_testing = (
            None,
            None,
            None,
            None,
        )
        self.prepare_training_data_from_csv_data(self.csv_data)

    @staticmethod
    def __shuffle_data(csv_data):
        return csv_data.reindex(np.random.permutation(csv_data.index))

    def prepare_training_data_from_csv_data(self, csv_data):
        csv_data = self.__shuffle_data(csv_data)
        # input data
        x_data = csv_data.drop(["Classification"], axis=1).values
        # output data
        y_data = csv_data["Classification"].values
        # split data values into training and test sets
        x_data_training, x_data_testing, self.y_data_training, self.y_data_testing = train_test_split(
            x_data, y_data, test_size=self.test_size, random_state=0
        )
        # normalize data values
        self.x_data_training, self.x_data_testing = self.__normalize_values(
            x_data_training, x_data_testing
        )

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

    def deep_belief_network_prediction(
        self, learning_rate, training_iterations, testing_iterations=10, hidden_layer_sizes_array=[10,10]
    ):
        accuracy_list = []
        for x in range(testing_iterations):
            self.prepare_training_data_from_csv_data(self.csv_data)
            classifier = SupervisedDBNClassification(
                hidden_layers_structure=hidden_layer_sizes_array,
                learning_rate_rbm=learning_rate / 2,
                learning_rate=learning_rate,
                n_epochs_rbm=int(training_iterations / 10),
                n_iter_backprop=training_iterations,
                batch_size=32,
                activation_function="relu",
                dropout_p=0.2,
            )
            classifier.fit(self.x_data_training, self.y_data_training)
            y_data_prediction = classifier.predict(self.x_data_testing)
            classifier_accuracy = accuracy_score(self.y_data_testing, y_data_prediction)
            accuracy_list.append(classifier_accuracy)
        return max(accuracy_list)

    def convolutional_neural_network_prediction(
        self, training_iterations, testing_iterations=10, hidden_layer_sizes_array=[10,10]
    ):
        accuracy_list = []
        for x in range(testing_iterations):
            self.prepare_training_data_from_csv_data(self.csv_data)
            # input layer + first hidden layer
            classifier = Sequential()
            classifier.add(
                Dense(
                    kernel_initializer="uniform",
                    activation="relu",
                    units=len(self.features),
                )
            )
            # hidden layers
            for hidden_layer_size in hidden_layer_sizes_array:
                classifier.add(
                    Dense(
                        units=hidden_layer_size, kernel_initializer="uniform", activation="relu"
                    )
                )
            # output layer
            classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))
            classifier.compile(
                optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
            )
            classifier.fit(
                self.x_data_training,
                self.y_data_training,
                batch_size=32,
                nb_epoch=training_iterations,
            )
            y_data_prediction = classifier.predict(self.x_data_testing)
            y_data_prediction = y_data_prediction > 0.5
            classifier_accuracy = accuracy_score(self.y_data_testing, y_data_prediction)
            accuracy_list.append(classifier_accuracy)
        return max(accuracy_list)

    def perceptron_neural_network_prediction(
        self, training_iterations, testing_iterations=10, hidden_layer_sizes_array=[10,10]
    ):
        accuracy_list = []
        for x in range(testing_iterations):
            self.prepare_training_data_from_csv_data(self.csv_data)
            mlp = MLPClassifier(
                hidden_layer_sizes=([layer_size for layer_size in hidden_layer_sizes_array]), max_iter=training_iterations
            )
            mlp.fit(self.x_data_training, self.y_data_training)
            predictions = mlp.predict(self.x_data_testing)
            classifier_accuracy = accuracy_score(self.y_data_testing, predictions)
            accuracy_list.append(classifier_accuracy)
        return max(accuracy_list)

    def write_stats_to_csv(self, fullpath, stats_dict_list):
        with open(file=fullpath, mode="w+") as stats_file:
            csv_writer = writer(stats_file, dialect=",")
            headers = list(stats_dict_list[0].keys())
            csv_writer.writerow(headers)
            for stats_dict in stats_dict_list:
                stats_dict_values = list(stats_dict.values())
                csv_writer.writerow(stats_dict_values)


nn = NeuralNetwork(csv_file_path="dataR2.csv", csv_file_test_data_size_in_percents=20)

training_iterations = 1
dbn_accuracy = nn.deep_belief_network_prediction(
    learning_rate=0.1, training_iterations=training_iterations
)
cnn_accuracy = nn.convolutional_neural_network_prediction(
    training_iterations=training_iterations
)
pnn_accuracy = nn.perceptron_neural_network_prediction(
    training_iterations=training_iterations
)

print(dbn_accuracy, cnn_accuracy, pnn_accuracy)
