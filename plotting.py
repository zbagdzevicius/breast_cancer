import matplotlib
from pandas import read_csv
import matplotlib.pyplot as plt

csv_file_path = "training.csv"
csv_data = read_csv(csv_file_path, delimiter=",")
# print(csv_data.loc[csv_data['training_iterations'] == 400])


def preprocessing_accuracy(
    csv_data, number_of_hidden_layers, neurons, training_iterations
):
    number_of_hidden_layers = f"(number_of_hidden_layers == 1)"
    neurons = f"(neurons == 30)"
    # training_iterations = f"(training_iterations == {training_iterations})"
    concat = " & "
    normalization_data = csv_data.query(
        f"(preprocessing == 'normalization'){concat}{number_of_hidden_layers}{concat}{neurons}"
    )
    standardization_data = csv_data.query(
        f"(preprocessing == 'standardization'){concat}{number_of_hidden_layers}{concat}{neurons}"
    )
    normalization_results = normalization_data[
        ["pnn_accuracy", "cnn_accuracy", "dbn_accuracy", "training_iterations"]
    ]
    print(normalization_data)
    standartization_results = standardization_data[
        ["pnn_accuracy", "cnn_accuracy", "dbn_accuracy", "training_iterations"]
    ]

    figure, axis = plt.subplots(nrows=1, ncols=2)
    print(figure, axis)
    axis[0].set_title('accuracy preprocessed with normalization')
    axis[0].set_ylabel('accuracy')
    axis[1].set_title('accuracy preprocessed with standardization')
    axis[1].set_ylabel('accuracy')
    axis[1].set_xlabel('accuracy')
    normalization_results.plot(
        kind="line", x="training_iterations", y="pnn_accuracy", ax=axis[0]
    )
    normalization_results.plot(
        kind="line", x="training_iterations", y="cnn_accuracy", ax=axis[0], color="g"
    )
    normalization_results.plot(
        kind="line", x="training_iterations", y="dbn_accuracy", ax=axis[0], color="r"
    )

    standartization_results.plot(
        kind="line", x="training_iterations", y="pnn_accuracy", ax=axis[1]
    )
    standartization_results.plot(
        kind="line", x="training_iterations", y="cnn_accuracy", ax=axis[1], color="g"
    )
    standartization_results.plot(
        kind="line", x="training_iterations", y="dbn_accuracy", ax=axis[1], color="r"
    )
    plt.show()


# def preprocessing_accuracy(csv_data, number_of_hidden_layers, neurons, training_iterations):
#     number_of_hidden_layers = f"(number_of_hidden_layers == '{number_of_hidden_layers}')"
#     neurons = f"(neurons == {neurons})"
#     training_iterations = f"(training_iterations == {training_iterations})"
#     concat = " & "
#     normalization_data = csv_data.query(f"(preprocessing == 'normalization'){concat}{number_of_hidden_layers}{concat}{neurons}{concat}{training_iterations}")
#     standardization_data = csv_data.query(f"(preprocessing == 'standardization'){concat}{number_of_hidden_layers}{concat}{neurons}{concat}{training_iterations}")
#     print(normalization_data[['pnn_accuracy','pnn_accuracy','dbn_accuracy','preprocessing']])
#     print(standardization_data[['pnn_accuracy','pnn_accuracy','dbn_accuracy','preprocessing']])


def training_accuracy(csv_data, number_of_hidden_layers, neurons, training_iterations):
    number_of_hidden_layers = (
        f"(number_of_hidden_layers == '{number_of_hidden_layers}')"
    )
    neurons = f"(neurons == {neurons})"
    training_iterations = f"(training_iterations == {training_iterations})"
    concat = " & "
    normalization_data = csv_data.query(
        f"(preprocessing == 'normalization'){concat}{number_of_hidden_layers}{concat}{neurons}{concat}{training_iterations}"
    )
    # normalization_data = csv_data.query(f"(preprocessing == 'normalization)'{concat}{number_of_hidden_layers}{concat}{neurons}{concat}{training_iterations}")
    # standardization_data = csv_data.query("preprocessing == 'standardization'")
    # print(normalization_data[['number_of_hidden_layers','preprocessing']])
    print(normalization_data)


def neurons_accuracy(csv_data, number_of_hidden_layers, neurons, training_iterations):
    number_of_hidden_layers = (
        f"(number_of_hidden_layers == '{number_of_hidden_layers}')"
    )
    neurons = f"(neurons == {neurons})"
    training_iterations = f"(training_iterations == {training_iterations})"
    concat = " & "
    normalization_data = csv_data.query(
        f"(preprocessing == 'normalization'){concat}{number_of_hidden_layers}{concat}{neurons}{concat}{training_iterations}"
    )
    # normalization_data = csv_data.query(f"(preprocessing == 'normalization)'{concat}{number_of_hidden_layers}{concat}{neurons}{concat}{training_iterations}")
    # standardization_data = csv_data.query("preprocessing == 'standardization'")
    # print(normalization_data[['number_of_hidden_layers','preprocessing']])
    print(normalization_data)


def training_time_of_hidden_layers(
    csv_data, number_of_hidden_layers, neurons, training_iterations
):
    number_of_hidden_layers = (
        f"(number_of_hidden_layers == '{number_of_hidden_layers}')"
    )
    neurons = f"(neurons == {neurons})"
    training_iterations = f"(training_iterations == {training_iterations})"
    concat = " & "
    normalization_data = csv_data.query(
        f"(preprocessing == 'normalization'){concat}{number_of_hidden_layers}{concat}{neurons}{concat}{training_iterations}"
    )
    # normalization_data = csv_data.query(f"(preprocessing == 'normalization)'{concat}{number_of_hidden_layers}{concat}{neurons}{concat}{training_iterations}")
    # standardization_data = csv_data.query("preprocessing == 'standardization'")
    # print(normalization_data[['number_of_hidden_layers','preprocessing']])
    print(normalization_data)


def training_time_of_neurons(
    csv_data, number_of_hidden_layers, neurons, training_iterations
):
    number_of_hidden_layers = (
        f"(number_of_hidden_layers == '{number_of_hidden_layers}')"
    )
    neurons = f"(neurons == {neurons})"
    training_iterations = f"(training_iterations == {training_iterations})"
    concat = " & "
    normalization_data = csv_data.query(
        f"(preprocessing == 'normalization'){concat}{number_of_hidden_layers}{concat}{neurons}{concat}{training_iterations}"
    )
    # normalization_data = csv_data.query(f"(preprocessing == 'normalization)'{concat}{number_of_hidden_layers}{concat}{neurons}{concat}{training_iterations}")
    # standardization_data = csv_data.query("preprocessing == 'standardization'")
    # print(normalization_data[['number_of_hidden_layers','preprocessing']])
    print(normalization_data)


def plot_density(csv_data):
    csv_data.plot(kind="density", subplots=True, layout=(3, 3), sharex=False)
    plt.show()


def plot_barh(csv_data):
    csv_data.plot(kind="barh", x="number_of_hidden_layers", y="pnn_accuracy")
    plt.show()


def plot_histogram(csv_data):
    csv_data.hist()
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
