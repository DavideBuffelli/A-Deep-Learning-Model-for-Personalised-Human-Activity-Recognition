import tensorflow as tf
import os
import csv
import numpy as np
from deepSense import deepSense_model_fn, input_fn
from datetime import datetime

"""
    TEST 01
    
    We are using the DeepSense model with the parameters taken from the code made available
    by the authors, and we are using leave-one-out cross validation (on the users) to determine
    the accuracy on the "augmented" dataset(120'000 elements) and on the "non-augmented" one
    (12'000 elements).
"""

# Directories containing the datasets.
DATASET_120_PATH = "/Path/To/Data/Dir"
DATASET_12_PATH = "/Path/To/Data/Dir"

# In the tests directory we create a result directory if it does not already exist.
TESTS_DIR = "/Path/To/Test/Dir"
RESULTS_DATA_DIR = os.path.join(TESTS_DIR, "results")
if not os.path.exists(RESULTS_DATA_DIR):
    os.mkdir(RESULTS_DATA_DIR)

# The results of the test will be written in a text file named test01.txt.
test01_results_filename = os.path.join(RESULTS_DATA_DIR, "test01.txt")


def cross_validation(data_folder_path, data_prefix, dataset_name, model_dir_path, model_params, output_file):
    users = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    batch_size = 64 # Size of the batch used for training and evaluation.
    
    accuracy_sum = 0 # We will use this two variables to keep track of the sum of the accuracies.
    mean_class_accuracy_sum = 0
    
    for user in users:
        print("--- User ", user)
        # We need a new model directory for each execution, otherwise we would be re-using
        # an already trained model.
        current_model_dir = os.path.join(model_dir_path, dataset_name + "-user-" + user)
        os.mkdir(current_model_dir)
        
        # Create the estimator.
        deepSense_classifier = tf.estimator.Estimator(
            model_fn = deepSense_model_fn,
            model_dir = current_model_dir,
            params = model_params)
        
        # Get the path of the data for training and evaluation of the current user.
        user_data_folder_path = os.path.join(data_folder_path, data_prefix + user)
        
        # Train the estimator with the training data(which is the data from all other users).
        print("Start training")
        training_data_folder_path = os.path.join(user_data_folder_path, "train")
        deepSense_classifier.train(lambda:input_fn(batch_size, True, training_data_folder_path))
        print("End training")
        
        # Evaluate accuracy on test set(which is the current users data).
        print("Start eval")
        eval_data_folder_path = os.path.join(user_data_folder_path, "eval")
        eval_result = deepSense_classifier.evaluate(lambda:input_fn(batch_size, False, eval_data_folder_path))
        print("Test Set Accuracy: {accuracy:0.3f}\nMean per Class Accuracy: {mean_perClass_accuracy:0.3f}".format(**eval_result))
        print("End eval")
        
        accuracy_sum += eval_result["accuracy"]
        mean_class_accuracy_sum += eval_result["mean_perClass_accuracy"]
        
        # Write the results on the file.
        with open(output_file, "a") as out_file:
            out_file.write("--- User " + user + "\n")
            out_file.write("Test Set Accuracy: {accuracy:0.3f}\nMean per Class Accuracy: {mean_perClass_accuracy:0.3f}\n".format(**eval_result))
            
        # In the model directory we are going to write a .csv file with the values of the
        # kernel and the bias of the output layer of the final trained model. This will be 
        # used for the initialization of the transfer learning model in test02.
        kernel = deepSense_classifier.get_variable_value("dense/kernel")
        bias = deepSense_classifier.get_variable_value("dense/bias")
        with open(os.path.join(current_model_dir, "kernel_bias.csv"), "w") as csvfile:
                csv_writer = csv.writer(csvfile)    
                csv_writer.writerow(np.concatenate((np.reshape(kernel, -1), bias)))
            
    # Now we can calculate the mean accuracy between all users...
    cross_validation_accuracy = accuracy_sum / 9
    cross_validation_mean_per_class_accuracy = mean_class_accuracy_sum / 9
    # And write it on the output file.
    with open(output_file, "a") as out_file:
        out_file.write("\n\nFinal Cross Validation Accuracy: " + str(cross_validation_accuracy))
        out_file.write("\nFinal Cross Validation Mean Per Class Accuracy: " + str(cross_validation_mean_per_class_accuracy))


def perform_test01():
    # Create a directory where we will save all the model directories.
    model_dir_path = os.path.join(TESTS_DIR, "modelDir") 
    if not os.path.exists(model_dir_path):
        os.mkdir(model_dir_path)
        
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

# ---------------------------------- Dataset 120'000 ----------------------------------
# We start with the bigger dataset.
    with open(test01_results_filename, "a") as out_file:
        out_file.write("\n\n\n-------------------------- TEST 01 - Date: " + str(datetime.now()) + " --------------------------\n")
        out_file.write("------------- Dataset 120'000 -------------\n")
    print("------------- Dataset 120'000 -------------")
    cross_validation(DATASET_120_PATH, "sepHARData_", "dataset120", model_dir_path, default_params, test01_results_filename)
        
# ---------------------------------- Dataset 12'000    ----------------------------------
# Same as before, but with the other dataset.
    with open(test01_results_filename, "a") as out_file:
        out_file.write("\n\n\n------------- Dataset 12'000 -------------\n")
    print("------------- Dataset 12'000 -------------")
    cross_validation(DATASET_12_PATH, "user_", "dataset12", model_dir_path, default_params, test01_results_filename)