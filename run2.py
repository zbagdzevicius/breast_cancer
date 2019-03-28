from sklearn.metrics.classification import accuracy_score
from sklearn.preprocessing import StandardScaler, scale, normalize
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from dbn import SupervisedDBNClassification
import numpy as np
from pandas import read_csv
from csv import writer
import time


class NeuralNetwork:
    def __init__(
        self,
        csv_file_path,
        csv_file_test_data_size_in_percents=75,
        preprocessing="normalization",
    ):
        self.preprocessing = preprocessing
        self.test_size = csv_file_test_data_size_in_percents / 100
        self.csv_data = read_csv(csv_file_path, delimiter=",")
        self.data_size = len(self.csv_data)
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
        if self.preprocessing == "standardization":
            self.x_data_training, self.x_data_testing = self.__standardize_values(
                x_data_training, x_data_testing
            )
        else:
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
    def __standardize_values(x_data_training, x_data_testing):
        sc = StandardScaler()
        sc.fit(x_data_training)
        x_data_training = sc.fit_transform(x_data_training)
        x_data_testing = sc.transform(x_data_testing)
        return x_data_training, x_data_testing

    @staticmethod
    def __normalize_values(x_data_training, x_data_testing):
        x_data_training = normalize(x_data_training)
        x_data_testing = normalize(x_data_testing)
        return x_data_training, x_data_testing

    def deep_belief_network_prediction(
        self,
        learning_rate,
        training_iterations,
        testing_iterations=10,
        hidden_layer_sizes_array=[10, 10],
    ):
        self.prepare_training_data_from_csv_data(self.csv_data)
        classifier = SupervisedDBNClassification(
            hidden_layers_structure=hidden_layer_sizes_array,
            learning_rate_rbm=learning_rate / 2,
            learning_rate=learning_rate,
            n_epochs_rbm=int(training_iterations / 10),
            n_iter_backprop=training_iterations,
            batch_size=256,
            activation_function="relu",
            dropout_p=0.2,
        )
        classifier

    def convolutional_neural_network_prediction(
        self,
        training_iterations,
        testing_iterations=10,
        hidden_layer_sizes_array=[10, 10],
    ):
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
                    units=hidden_layer_size,
                    kernel_initializer="uniform",
                    activation="relu",
                )
            )
        # output layer
        classifier.add(
            Dense(units=1, kernel_initializer="uniform", activation="sigmoid")
        )
        classifier.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        # classifier.fit(
        #     self.x_data_training,
        #     self.y_data_training,
        #     batch_size=256,
        #     epochs=training_iterations,
        # )
        print(classifier.metrics['accuracy'])
        # y_data_prediction = classifier.predict(self.x_data_testing)
        # y_data_prediction = y_data_prediction > 0.5
        # classifier_accuracy = accuracy_score(self.y_data_testing, y_data_prediction)

    def perceptron_neural_network_prediction(
        self,
        training_iterations,
        testing_iterations=10,
        hidden_layer_sizes_array=[10, 10],
    ):
        self.prepare_training_data_from_csv_data(self.csv_data)
        mlp = MLPClassifier(
            hidden_layer_sizes=(
                [layer_size for layer_size in hidden_layer_sizes_array]
            ),
            max_iter=training_iterations,
        )
        mlp.fit(self.x_data_training, self.y_data_training)
        return mlp.loss_curve_
        # predictions = mlp.predict(self.x_data_testing)
        # classifier_accuracy = accuracy_score(self.y_data_testing, predictions)

    def write_stats_to_csv(self, fullpath, stats_dict_list):
        with open(file=fullpath, mode="a+") as stats_file:
            csv_writer = writer(stats_file)
            stats_dict_values = list(stats_dict_list.values())
            csv_writer.writerow(stats_dict_values)


nn = NeuralNetwork(
    csv_file_path="dataR2.csv",
    csv_file_test_data_size_in_percents=25,
    preprocessing="standardization",
)

for preprocessing_type in ["normalization", "standardization"]:
    nn = NeuralNetwork(
        csv_file_path="dataR2.csv",
        csv_file_test_data_size_in_percents=25,
        preprocessing=preprocessing_type,
    )
    for neuron_alpha in range(1, 8):
        neurons = int(
            nn.data_size / (len(nn.features) + len(nn.label)) * 2 * neuron_alpha
        )

        for hidden_layers_sizes in range(1, 3):
            hidden_layers = []
            for hidden_layers_size in range(hidden_layers_sizes):
                hidden_layers.append(neurons)
            
            start = time.time()
            pnn_accuracies = nn.perceptron_neural_network_prediction(
                training_iterations=4000,
                hidden_layer_sizes_array=hidden_layers,
                testing_iterations=1,
            )
            end = time.time()
            pnn_training_time = end - start

            for accuracy_index in range(len(pnn_accuracies)):
                pnn_accuracy = pnn_accuracies[accuracy_index]

                nn.write_stats_to_csv(
                    "perceptron_performance.csv",
                    {
                        "number_of_hidden_layers": hidden_layers_sizes,
                        "preprocessing": preprocessing_type,
                        "neurons": neurons,
                        "training_iterations": accuracy_index+1,
                        "cost": pnn_accuracy,
                        "pnn_training_time": pnn_training_time,
                    },
                )
