import numpy as np 
import pandas as pd 
import pickle as pkl 
import math
from random import randint
import matplotlib.pyplot as plt
import sys
import os
from PIL import Image

__set_file = "images.csv"

__training_percent = 75

__params = ["image_name"]

__output = ["Output"]

__neural_network_model = "neural_network_model.pickle"

__images_folder = "handwritten_digit_corpus"

__number_mapping = {
	0: [1,0,0,0,0,0,0,0,0,0],
	1: [0,1,0,0,0,0,0,0,0,0],
	2: [0,0,1,0,0,0,0,0,0,0],
	3: [0,0,0,1,0,0,0,0,0,0],
	4: [0,0,0,0,1,0,0,0,0,0],
	5: [0,0,0,0,0,1,0,0,0,0],
	6: [0,0,0,0,0,0,1,0,0,0],
	7: [0,0,0,0,0,0,0,1,0,0],
	8: [0,0,0,0,0,0,0,0,1,0],
	9: [0,0,0,0,0,0,0,0,0,1]
}

__neural_network = {
	"layer_1": {
		"number_of_neurons": 16,
		"number_of_inputs": 784
	},
	"layer_2":{
		"number_of_neurons": 12,
		"number_of_inputs": 16
	},
	"layer_3":{
		"number_of_neurons": 10,
		"number_of_inputs": 12
	},
	"layers_count": 3
}

def sigmoid(input_vec):
	return 1/(1 + np.exp(-input_vec))

def sigmoid_diff(input_vec):
	return sigmoid(input_vec) * sigmoid(1 - input_vec)

def summation(weights, input_vec, bias_vec):
	intermediate_vec = np.matmul(weights, input_vec) + bias_vec
	return intermediate_vec

def initialize_weights(number_of_inputs, neurons_count):
	return np.random.randn(neurons_count, number_of_inputs)

def initialize_bias(neurons_count):
	return np.random.randn(neurons_count, 1)

def get_last_index(length_of_df):
	return math.floor(length_of_df * (__training_percent/100))

def calculate_error(actual_output, desired_output, last_neuron_layer_count):
	diff_error = actual_output - desired_output
	squared_error = diff_error ** 2
	squared_sum_error = np.sum(squared_error)
	avg_squared_sum_error = squared_sum_error / last_neuron_layer_count
	return avg_squared_sum_error

def error_derivative(actual_output, desired_output, last_neuron_layer_count):
	return ((actual_output - desired_output)/last_neuron_layer_count)

def store_model(model_perceptron):
	with open(__neural_network_model, 'wb') as handle:
		pkl.dump(model_perceptron, handle, protocol=pkl.HIGHEST_PROTOCOL)

def load_model():
	with open(__neural_network_model, 'rb') as handle:
		model_perceptron = pkl.load(handle)
	return model_perceptron

def get_data_set(set_type="train"):
	dataframe = pd.read_csv(__set_file)
	
	dataframe["Output"] = dataframe[["label"]].apply(lambda x: np.array([__number_mapping.get(x.to_frame().T.reset_index().label[0])]).T, axis=1)
	# dataframe = dataframe.sample(frac=1).reset_index(drop=True)

	__last_idx = get_last_index(len(dataframe))

	if set_type == "train":
		chuncked_dataframe = dataframe[:__last_idx]
	elif set_type == "test":
		chuncked_dataframe = dataframe[__last_idx:]

	return chuncked_dataframe[__params].as_matrix(), chuncked_dataframe[__output].as_matrix()

def confusion_matrix(actual_output, desired_output):
	y_actu = pd.Series(actual_output, name='Actual')
	y_pred = pd.Series(desired_output, name='Predicted')
	df_confusion = pd.crosstab(y_actu, y_pred)

	return df_confusion

def get_output_value(output_vector):
	return (output_vector.index(max(output_vector)) + 1)

def get_input_image_as_vec(image_file_name):
	file_path = os.path.join(__images_folder, image_file_name)
	image_buffer = Image.open(file_path).convert('L')
	image_array = np.array(image_buffer)
	flattened_array = image_array.ravel()
	return flattened_array

def train_nn():
	input_params, desired_output = get_data_set("train")
	forward_neuron_layer_order = list(range(1, __neural_network.get("layers_count") + 1))
	backward_neuron_layer_order = forward_neurons_order[::-1]

	# INITIALIZATION OF WEIGHTS AND BIAS
	# IT IS REQUIRED BEFORE TRAINING OF NEURAL NETWORK
	for each_layer in forward_neuron_layer_order:
		layer_name = "_".join(["layer", each_layer])
		__neural_network[layer_name]["weights"] = initialize_weights(__neural_network.get(layer_name).get("number_of_inputs"), 
									 								 __neural_network.get(layer_name).get("number_of_neurons"))
		__neural_network[layer_name]["bias"] = initialize_bias(__neural_network.get(layer_name).get("number_of_neurons"))

	# TODO:
	# STOCHASTIC GRADIENT DESCENT WOULD BE USED,
	# BECAUSE DATASET IS TOO LARGE FOR NORMAL TRAINING
	# THUS IT NEEDS TO BE LEARNT FIRST
	__learning_rate = 0.2

	last_layer = "_".join(["layer", model_neural_network.get("layers_count")])

	__n_times = 5
# 	__batch_size = 100
	__total_dataset_size = input_params.shape[0]
	__iteration_times = __n_times * math.ceil(__total_dataset_size)

	__start_idx = 0
	__end_idx = __total_dataset_size

	__errors = []

	for each_time in range(__iteration_times):
		# UNCOMMENT THIS OUT FOR BATCH SIZED GRADIENT DESCENT
		# for each_time in range(__batch_size):
		# FEEDFORWARD NETWORK
		random_index = randint(__start_idx, __end_idx)
		input_vec = np.array([get_input_image_as_vec(input_params[random_index][0])])
		desired_output_vec = np.array([desired_output[random_index][0]])

		# FEEDING FORWARD STARTS HERE
		# FOR EACH NEURON LAYERS
		__net_output = []
		__net_input = []

		# FEED FORWARDING
		for each_layer in forward_neuron_layer_order:
			layer_name = "_".join(["layer", each_layer])
			# INPUT FOR EACH LAYER IN INDEXING ORDER
			__net_input.append(input_vec)
			net_output = summation(
				__neural_network[layer_name]["weights"],
				input_vec.T,
				__neural_network[layer_name]["bias"]
			)
			# OUTPUT FOR EACH LAYER IN INDEXING ORDER
			__net_output.append(net_output)
			synaptic_output = sigmoid(net_output)

			# ASSIGNING SYNAPTIC OUTPUT TO INPUT
			input_vec = synaptic_output

		# BACKWARD PROPAGATION
		delta_in_weights = []
		delta_in_bias = []

		for each_layer in backward_neuron_layer_order:
			layer_name = "_".join(["layer", each_layer])
			if layer_name == last_layer:
				error_to_backpropagate = error_derivative(synaptic_output, desired_output_vec, 
					__neural_network.get(layer_name).get("number_of_neurons"))
			else:
				prior_layer = "_".join(["layer", each_layer + 1])
				error_to_backpropagate = np.matmul(__neural_network[prior_layer]["weights"],
					error_to_backpropagate.T)
			# MINIMIZING ERROR
			derivative_of_sig = sigmoid_diff(__net_output[each_layer])
			input_vec = __net_input[each_layer]

			error_to_backpropagate = np.multiply(error_to_backpropagate, derivative_of_sig)
			weights_delta = np.matmul(input_vec.T, error_to_backpropagate.T).T
			bais_delta = error_to_backpropagate

			delta_in_weights.append(weights_delta)
			delta_in_bias.append(bais_delta)
			
		# UPDATE WEIGHTS AFTER
		# EACH ITERATION
		for each_layer in backward_neuron_layer_order:
			layer_name = "_".join(["layer", each_layer])				
			__neural_network[layer_name]["weights"] -= __learning_rate * delta_in_weights[each_layer]
			__neural_network[layer_name]["bias"] -= __learning_rate * delta_in_bias[each_layer]

		calculated_error = calculate_error(synaptic_output, desired_output_vec, __neural_network.get(last_layer).get("number_of_neurons"))
		__errors.append(calculated_error)

	plt.plot(__errors)
	plt.ylabel('Error plotting')
	plt.show()

	return __neural_network

def test_nn(model_neural_network):
	__error_threshold = 0.01

	test_params, desired_output = get_data_set("test")

	test_dataset_size = len(test_params)

	calculated_output_arr = []
	desired_output_arr = []
	correct_classification = 0

	last_layer = "_".join(["layer", model_neural_network.get("layers_count")])

	forward_neuron_layer_order = list(range(1, model_neural_network.get("layers_count") + 1))

	for idx in range(test_dataset_size):
		# FEED FORWARDING
		desired_output_vec = np.array([desired_output[idx][0]])
		input_vec = np.array([get_input_image_as_vec(input_params[random_index][0])])

		for each_layer in forward_neuron_layer_order:
			layer_name = "_".join(["layer", each_layer])
			# INPUT FOR EACH LAYER IN INDEXING ORDER
			__net_input.append(input_vec)
			net_output = summation(
				model_neural_network[layer_name]["weights"],
				input_vec.T,
				model_neural_network[layer_name]["bias"]
			)
			# OUTPUT FOR EACH LAYER IN INDEXING ORDER
			__net_output.append(net_output)
			synaptic_output = sigmoid(net_output)

			# ASSIGNING SYNAPTIC OUTPUT TO INPUT
			input_vec = synaptic_output

		calculated_error = calculate_error(synaptic_output, desired_output_vec, model_neural_network.get(last_layer).get("number_of_neurons"))

		calculated_output = get_output_value(desired_output_vec)
		desired_output = get_output_value(synaptic_output)

		if calculated_output == desired_output:
			correct_classification += 1

		calculated_output_arr.append(get_output_value(calculated_output))
		desired_output_arr.append(get_output_value(desired_output))

	accuracy = (correct_classification / test_dataset_size) * 100.0
	print("Model accuracy: ",accuracy)

	return confusion_matrix(calculated_output_arr, desired_output_arr)

if __name__ == "__main__":
	try:
		command = sys.argv[1]
	except Exception as ex:
		command = "empty"

	if command == "train":
		model = train_nn()
		store_model(model)
	elif command == "test":
		model = load_model()
		result = test_nn(model)
		print("-- Confusion matrix --")
		print(result)
	else:
		print("Please provide command")
		print("Example: python perceptron.py train")
		print("Example: python perceptron.py test")
