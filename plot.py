import matplotlib
from pandas import read_csv
import matplotlib.pyplot as plt

csv_file_path = "perceptron_performance.csv"
csv_data = read_csv(csv_file_path, delimiter=",")
# print(csv_data.loc[csv_data['training_iterations'] == 400])


def preprocessing_accuracy(
    csv_data, number_of_hidden_layers, neurons, training_iterations
):
    number_of_hidden_layers = f"(number_of_hidden_layers == 1)"
    neurons = f"(neurons == 30)"
    concat = " & "
    normalization_data = csv_data.query(
        f"(preprocessing == 'normalization'){concat}{number_of_hidden_layers}{concat}{neurons}"
    )
    standardization_data = csv_data.query(
        f"(preprocessing == 'standardization'){concat}{number_of_hidden_layers}{concat}{neurons}"
    )
    normalization_results = normalization_data[
        ["cost", "training_iterations"]
    ]
    print(normalization_data)
    standartization_results = standardization_data[
        ["cost", "training_iterations"]
    ]

    figure, axis = plt.subplots(nrows=1, ncols=2)
    print(figure, axis)
    axis[0].set_title('cost when preprocessed with normalization')
    axis[0].set_ylabel('cost')
    axis[1].set_title('cost when preprocessed with standardization')
    axis[1].set_ylabel('cost')
    normalization_results.plot(
        kind="line", x="training_iterations", y="cost", ax=axis[0]
    )

    standartization_results.plot(
        kind="line", x="training_iterations", y="cost", ax=axis[1]
    )
    plt.show()


number_of_hidden_layers = [1, 2]
neurons = [20, 30, 40]
training_iterations = [400, 500, 600, 700, 800]
preprocessing_accuracy(
    csv_data=csv_data,
    number_of_hidden_layers=number_of_hidden_layers[0],
    neurons=neurons[0],
    training_iterations=training_iterations[0],
)
# plot_barh(csv_data)
