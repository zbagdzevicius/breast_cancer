import numpy as np
from pandas import read_csv


class NeuralNetwork:
    def __init__(self, csv_data_path,
                 csv_data_training_size_in_percents=75,
                 is_shuffled_data_required=True,
                 training_iterations=200,
                 learning_rate=0.1):
        csv_data = read_csv(csv_data_path, delimiter=',')
        if is_shuffled_data_required:
            csv_data = self.shuffle_data(csv_data)

        size_training = csv_data_training_size_in_percents/100
        csv_data_headers = [header for header in csv_data]
        self.features = csv_data_headers[:-1]
        self.label = csv_data_headers[-1]


        data = csv_data.to_numpy(dtype=list)
        # input 
        x_data = np.array(data[:, :-1])
        x_data_training, x_data_testing = self.split_data_to_training_testing_sets(x_data, size_training)
        # output
        y_data = np.array(data[:, -1])
        y_data_training, y_data_testing = self.split_data_to_training_testing_sets(y_data, size_training)

        # self.output(training_iterations, learning_rate,x_data_output,y_data_input)

    @staticmethod
    def shuffle_data(data):
        return data.reindex(np.random.permutation(data.index))

    @staticmethod
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    @staticmethod
    def derivatives_sigmoid(x):
        return x * (1 - x)
    
    @staticmethod
    def split_data_to_training_testing_sets(data, size_training):
        return np.split(data, [int(size_training*len(data))])

    
    def output(self, training_iterations, learning_rate, X, y):
        number_of_input_layer_neurons = len(self.label)
        number_of_hidden_layer_neurons = 3
        number_of_output_neurons = len(self.features)

        weights=np.random.uniform(size=(number_of_input_layer_neurons,number_of_hidden_layer_neurons))
        bias=np.random.uniform(size=(1,number_of_hidden_layer_neurons))
        weights_out=np.random.uniform(size=(number_of_hidden_layer_neurons,number_of_output_neurons))
        bias_out=np.random.uniform(size=(1,number_of_output_neurons))
        for i in range(training_iterations):
            #Forward Propogation
            hidden_layer_input1=np.dot(X,weights)
            hidden_layer_input=hidden_layer_input1 + bias
            hiddenlayer_activations = self.sigmoid(hidden_layer_input)
            output_layer_input1=np.dot(hiddenlayer_activations,weights_out)
            output_layer_input= output_layer_input1+ bias_out
            output = self.sigmoid(output_layer_input)
            #Backpropagation
            E = y-output
            slope_output_layer = self.derivatives_sigmoid(output)
            slope_hidden_layer = self.derivatives_sigmoid(hiddenlayer_activations)
            d_output = E * slope_output_layer
            Error_at_hidden_layer = d_output.dot(weights_out.T)
            d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
            weights_out += hiddenlayer_activations.T.dot(d_output) *learning_rate
            bias_out += np.sum(d_output, axis=0,keepdims=True) *learning_rate
            weights += X.T.dot(d_hiddenlayer) *learning_rate
            bias += np.sum(d_hiddenlayer, axis=0,keepdims=True) *learning_rate
        print(output)


nn = NeuralNetwork(csv_data_path="dataR2.csv")