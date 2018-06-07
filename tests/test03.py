import tensorflow as tf
import os
from shutil import copy
import csv
import numpy as np
from deepSense import deepSense_model_fn, input_fn, predict_input_fn
from transferLearning import tl_model_fn, tl_input_fn
from datetime import datetime
import random

# Directories containing the datasets.
DATASET_130_PATH = "/Users/davidebuffelli/Desktop/Data"

# In the tests directory we create a result directory if it does not already exist.
TESTS_DIR = "/Users/davidebuffelli/Desktop/Prova/tests"
RESULTS_DATA_DIR = os.path.join(TESTS_DIR, "results")
if not os.path.exists(RESULTS_DATA_DIR):
	os.mkdir(RESULTS_DATA_DIR)

# The results of the test will be written in a text file named test01.txt.
test03_results_filename = os.path.join(RESULTS_DATA_DIR, "test03.txt")


# Randomly permute the labels of training and validation examples of a user.
# DATA_DIRECTORY_PATH is the path to the folder containing the data for a user.
def permuteLabels(DATA_DIRECTORY_PATH):
	train_data_path = os.path.join(DATA_DIRECTORY_PATH, "train")
	eval_data_path = os.path.join(DATA_DIRECTORY_PATH, "eval")

	features = []
	labels = []

	# Extract all features and labels of the training examples.
	for file in os.listdir(train_data_path):
		file_path = os.path.join(train_data_path, file) 
		with open(file_path, "r") as csvfile:
						csv_reader = csv.reader(csvfile)
						line = next(csv_reader)
						l = list(map(float, line)) # Doubles require a lot less space than strings
                        features.append(l[:-6])
                        labels.append(l[-6:])

	# Shuffle the labels.
	random.shuffle(labels)

	# Put toghether the features with the shuffled labels and write the "new" dataset in the shuffle_train directory
	os.mkdir(os.path.join(DATA_DIRECTORY_PATH, "shuffle_train"))
	for i, f, l in zip(range(len(features)), features, labels):
		file_path = os.path.join(DATA_DIRECTORY_PATH, "shuffle_train", "train_" + str(i) + ".csv") 
		with open(file_path, "w") as csvfile:
						csv_writer = csv.writer(csvfile)
						csv_writer.writerow(f + l)
	
					
	# Now we do the same thing but for the validation data				
	features = []
	labels = []

	# Extract all features and labels of the validation examples.
	for file in os.listdir(eval_data_path):
		file_path = os.path.join(eval_data_path, file) 
		with open(file_path, "r") as csvfile:
						csv_reader = csv.reader(csvfile)
						line = next(csv_reader)
						l = list(map(float, line)) # Doubles require a lot less space than strings
                        features.append(l[:-6])
                        labels.append(l[-6:])

	# Shuffle the labels.
	random.shuffle(labels)

	# Put toghether the features with the shuffled labels and write the "new" dataset in the shuffle_eval directory
	os.mkdir(os.path.join(DATA_DIRECTORY_PATH, "shuffle_eval"))
	for i, f, l in zip(range(len(features)), features, labels):
		file_path = os.path.join(DATA_DIRECTORY_PATH, "shuffle_eval", "eval_" + str(i) + ".csv") 
		with open(file_path, "w") as csvfile:
						csv_writer = csv.writer(csvfile)
						csv_writer.writerow(f + l)

def perform_test03():	
	# Create a directory where we will save all the model directories.
	model_dir_path = os.path.join(TESTS_DIR, "modelDir") 
	if not os.path.exists(model_dir_path):
		os.mkdir(model_dir_path)
			
	# ------------------------------------- Part 1 -------------------------------------
	with open(test03_results_filename, "a") as out_file:
		out_file.write("\n\n\n-------------------------- TEST 03 - Date: " + str(datetime.now()) + " --------------------------\n")
		out_file.write("\n------ Training and Evaluating Full DeepSense Model on Shuffled Data ------\n")
		
	users = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
	batch_size = 64 # Batch size used for training and evaluation of the estimator
	accuracy_sum = 0 # We will use this two variables to keep track of the sum of the accuracies.
	mean_class_accuracy_sum = 0
	kb_dict = {} # We will save the kernel and biases needed to initialize the transfer leraning model in a dictionary.
	for user in users:
		# We are using the parameters of the model made available by the authors of DeepSense.
		default_params = {
				"acc_conv1_num_filters": 64,
				"acc_conv1_kernel_size": [1, 6*3],
				"acc_conv1_stride": [1, 6],
				"acc_conv1_padding": "VALID",
				"acc_conv1_dropout_rate": 0.2,
		
				"acc_conv2_num_filters": 64,
				"acc_conv2_kernel_size": [1, 3],
				"acc_conv2_stride": [1, 1],
				"acc_conv2_padding": "VALID",
				"acc_conv2_dropout_rate": 0.2,
		
				"acc_conv3_num_filters": 64,
				"acc_conv3_kernel_size": [1, 3],
				"acc_conv3_stride": [1, 1],
				"acc_conv3_padding": "VALID",
		
				"gyro_conv1_num_filters": 64,
				"gyro_conv1_kernel_size": [1, 6*3],
				"gyro_conv1_stride": [1, 6],
				"gyro_conv1_padding": "VALID",
				"gyro_conv1_dropout_rate": 0.2,
		
				"gyro_conv2_num_filters": 64,
				"gyro_conv2_kernel_size": [1, 3],
				"gyro_conv2_stride": [1, 1],
				"gyro_conv2_padding": "VALID",
				"gyro_conv2_dropout_rate": 0.2,
		
				"gyro_conv3_num_filters": 64,
				"gyro_conv3_kernel_size": [1, 3],
				"gyro_conv3_stride": [1, 1],
				"gyro_conv3_padding": "VALID",
		
				"sensor_conv_in_dropout_rate": 0.2,
		
				"sensor_conv1_num_filters": 64,
				"sensor_conv1_kernel_size": [1, 2, 8],
				"sensor_conv1_stride": [1, 1, 1],
				"sensor_conv1_padding": "SAME",
				"sensor_conv1_dropout_rate": 0.2,
		
				"sensor_conv2_num_filters": 64,
				"sensor_conv2_kernel_size": [1, 2, 6],
				"sensor_conv2_stride": [1, 1, 1],
				"sensor_conv2_padding": "SAME",
				"sensor_conv2_dropout_rate": 0.2,
		
				"sensor_conv3_num_filters": 64,
				"sensor_conv3_kernel_size": [1, 2, 4],
				"sensor_conv3_stride": [1, 1, 1],
				"sensor_conv3_padding": "SAME",
		
				"gru_cell1_dropout_output_keep_prob": 0.5,
				"gru_cell2_dropout_output_keep_prob": 0.5}
			
		print("--- User ", user)
		# Create the model directory
		current_model_dir = os.path.join(model_dir_path, "shuffle-user-" + user)
		os.mkdir(current_model_dir)
		
		# Create the estimator.
		deepSense_classifier = tf.estimator.Estimator(
			model_fn = deepSense_model_fn,
			model_dir = current_model_dir,
			params = default_params)
		
		# Get the path of the data for training and evaluation of the current user.
		user_data_folder_path = os.path.join(DATASET_130_PATH, "sepHARData_" + user)
		
		# Shuffle the labels of the examples.
		print("Shuffling data..")
		permuteLabels(user_data_folder_path)
		print("End Shuffle")
		
		# Train the estimator with the shuffled training data.
		print("Start training")
		training_data_folder_path = os.path.join(user_data_folder_path, "shuffle_train")
		deepSense_classifier.train(lambda:input_fn(batch_size, True, training_data_folder_path))
		print("End training")
		
		# Evaluate accuracy on shuffled test set.
		print("Start eval")
		eval_data_folder_path = os.path.join(user_data_folder_path, "shuffle_eval")
		eval_result = deepSense_classifier.evaluate(lambda:input_fn(batch_size, False, eval_data_folder_path))
		print("Test Set Accuracy: {accuracy:0.3f}\nMean per Class Accuracy: {mean_perClass_accuracy:0.3f}".format(**eval_result))
		print("End eval")
		
		accuracy_sum += eval_result["accuracy"]
		mean_class_accuracy_sum += eval_result["mean_perClass_accuracy"]
		
		# Write the results on the file.
		with open(test03_results_filename, "a") as out_file:
			out_file.write("--- User " + user + "\n")
			out_file.write("Test Set Accuracy: {accuracy:0.3f}\nMean per Class Accuracy: {mean_perClass_accuracy:0.3f}\n".format(**eval_result))
			
		# Save kernel and bias of training model. They will be 
		# used for initializing the transfer learning model.
		kernel = deepSense_classifier.get_variable_value("dense/kernel")
		bias = deepSense_classifier.get_variable_value("dense/bias")
		kb_dict[user] = (kernel, bias)
			
	# Now we can calculate the mean accuracy between all users...
	cross_validation_accuracy = accuracy_sum / 9
	cross_validation_mean_per_class_accuracy = mean_class_accuracy_sum / 9
	# And write it on the results file.
	with open(test03_results_filename, "a") as out_file:
		out_file.write("\n\nFinal Cross Validation Accuracy: " + str(cross_validation_accuracy))
		out_file.write("\nFinal Cross Validation Mean Per Class Accuracy: " + str(cross_validation_mean_per_class_accuracy))
	
	# ------------------------------------- Part 2 -------------------------------------	
	with open(test03_results_filename, "a") as out_file:
		out_file.write("\n\n\n------ Training and Evaluating Costum DeepSense Model on Regular Data ------\n")
	
	accuracy_sum = 0 # We will use this two variables to keep track of the sum of the accuracies.
	mean_class_accuracy_sum = 0
	for user in users:
		print("--- User ", user)
		# Create model directory
		current_model_dir = os.path.join(model_dir_path, "tl-shuffle-user-" + user)
		os.mkdir(current_model_dir)
		
		# Get the path of the data for training and evaluation of the current user.
		user_data_folder_path = os.path.join(DATASET_130_PATH, "tl_sepHARData_" + user)
		train_data_folder_path = os.path.join(user_data_folder_path, "train")
		eval_data_folder_path = os.path.join(user_data_folder_path, "eval")
		
		# Get the values for kernel and bias.
		kernel, bias = kb_dict[user]
			
		# Create the transfer learning estimator.
		deepSense_transferLearning_classifier = tf.estimator.Estimator(
			model_fn = tl_model_fn,
			model_dir = current_model_dir,
			params = {"kernel_value": kernel, 
						"bias_value": bias})
	
		print("Start Training")
		# Train the classifier on training set.
		deepSense_transferLearning_classifier.train(lambda:tl_input_fn(train_data_folder_path))
		print("End Training")
		
		print("Start Evaluation")
		# Evaluate accuracy on test set.
		eval_result = deepSense_transferLearning_classifier.evaluate(lambda:tl_input_fn(eval_data_folder_path))
		print("Test Set Accuracy: {accuracy:0.3f}\nMean per Class Accuracy: {mean_perClass_accuracy:0.3f}".format(**eval_result))
		print("End Evaluation")
		
		accuracy_sum += eval_result["accuracy"]
		mean_class_accuracy_sum += eval_result["mean_perClass_accuracy"]
		
		# Write user results on file.
		with open(test03_results_filename, "a") as out_file:
			out_file.write("--- User " + user + "\n")
			out_file.write("Test Set Accuracy: {accuracy:0.3f}\nMean per Class Accuracy: {mean_perClass_accuracy:0.3f}\n".format(**eval_result))
			
	# Now we can calculate the mean accuracy between all users...
	cross_validation_accuracy = accuracy_sum / 9
	cross_validation_mean_per_class_accuracy = mean_class_accuracy_sum / 9
	# And write it on the output file.
	with open(test03_results_filename, "a") as out_file:
		out_file.write("\n\nFinal Cross Validation Accuracy: " + str(cross_validation_accuracy))
		out_file.write("\nFinal Cross Validation Mean Per Class Accuracy: " + str(cross_validation_mean_per_class_accuracy))