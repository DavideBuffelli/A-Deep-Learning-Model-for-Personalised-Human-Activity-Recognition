import tensorflow as tf
import os
from shutil import copy
import csv
import numpy as np
from deepSense import deepSense_model_fn, input_fn, predict_input_fn
from transferLearning import tl_model_fn, tl_input_fn
from datetime import datetime

"""
    TEST 02
    
    This test performs leave-one-out cross validation evaluating the performances of the
    customized DeepSense framework.
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
test02_results_filename = os.path.join(RESULTS_DATA_DIR, "test02.txt")

# We will create a new dataset for the transfer learning model. The structure will be the same
# of the datasets for DeepSense: each file .csv file will contain a single row with 126 values.
# The first 120 represent the input for the transfer learning model (and are obtained from the 
# trained DeepSense models) and the last 6 represent the label of the ground truth.
def create_dataset(dataset_path, user_folder_prefix, model_dirs):
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
        
    users = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    activities = {0: "bike", 1: "sit", 2: "stand", 3: "walk", 4: "stairsup", 5: "stairsdown"}
    for user in users:
        print("Creating data for user ", user)
        # Get the path of the data for training and evaluation of the current user.
        user_data_folder_path = os.path.join(dataset_path, user_folder_prefix + user)
        eval_data_folder_path = os.path.join(user_data_folder_path, "eval")
        
        # Create folder where we will put the data.
        output_data_folder_path = os.path.join(dataset_path, "tl_" + user_folder_prefix + user)
        os.mkdir(output_data_folder_path)
        for i in range(6):
            os.mkdir(os.path.join(output_data_folder_path, activities[i]))
        
        # Create the DeepSense estimator. We need to use the model directory of a trained deepsense model
        # so we can get the value of the input to the output layer for each example.
        deepSense_classifier = tf.estimator.Estimator(
            model_fn = deepSense_model_fn,
            model_dir = model_dirs[user], 
            params = default_params)
    
        filelist = os.listdir(eval_data_folder_path)
        for file in filelist:
            current_file_path = os.path.join(eval_data_folder_path, file)
            label = None
            with open(current_file_path, "r") as csvfile:
                csv_reader = csv.reader(csvfile)
                line = next(csv_reader)
                label = np.array(line[-6:])
        
            # Obtain the input for the transfer learning model.
            predictions = deepSense_classifier.predict(lambda:predict_input_fn(current_file_path))
            pred = next(predictions)
            tl_input = pred["tl_input"]
            
            # Write a new file with tl_input and label. (The file will have the same name
            # as the original so we know which one generated each element of this dataset).
            # We first organize the examples putting them in a separate directory for each activity...
            output_folder_path = os.path.join(output_data_folder_path, activities[np.argmax(label)])
            output_filename = os.path.join(output_folder_path, file)
            with open(output_filename, "w") as csvfile:
                csv_writer = csv.writer(csvfile)    
                csv_writer.writerow(np.concatenate((tl_input, label)))
                
        # ...Then for each activity we divide the data in two halfes (one for training, one for evaluation).
        for i in range(6):
            activity_directory_path = os.path.join(output_data_folder_path, activities[i])
            filelist = os.listdir(activity_directory_path)
            number_of_elements = len(filelist)
            training_files = filelist[:number_of_elements // 2]
            eval_files = filelist[number_of_elements // 2 :]
            
            train_folder_path = os.path.join(output_data_folder_path, "train")
            if not os.path.exists(train_folder_path):
                os.mkdir(train_folder_path)
            eval_folder_path = os.path.join(output_data_folder_path, "eval")
            if not os.path.exists(eval_folder_path):
                os.mkdir(eval_folder_path)
            
            for file in training_files:
                copy(os.path.join(activity_directory_path, file), train_folder_path)
            for file in eval_files:
                copy(os.path.join(activity_directory_path, file), eval_folder_path)
        
        # Finally we also copy the kernel_bis file.
        copy(os.path.join(model_dirs[user], "kernel_bias.csv"), output_data_folder_path)
            
def cross_validation(data_folder_path, data_prefix, dataset_name, model_dir_path, output_file):        
    users = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]        
    
    accuracy_sum = 0 # We will use this two variables to keep track of the sum of the accuracies.
    mean_class_accuracy_sum = 0
    
    for user in users:
        print("--- User ", user)
        # We need a new model directory for each execution, otherwise we would be re-using
        # an already trained model.
        current_model_dir = os.path.join(model_dir_path, "tl-" + dataset_name + "-user-" + user)
        os.mkdir(current_model_dir)
        
        # Get the path of the data for training and evaluation of the current user.
        user_data_folder_path = os.path.join(data_folder_path, "tl_" + data_prefix + user)
        train_data_folder_path = os.path.join(user_data_folder_path, "train")
        eval_data_folder_path = os.path.join(user_data_folder_path, "eval")
        
        # Read the values for kernel and bias.
        kernel = None
        bias = None
        with open(os.path.join(user_data_folder_path, "kernel_bias.csv"), "r") as csvfile:
                csv_reader = csv.reader(csvfile)
                line = next(csv_reader)
                kernel = np.array([float(i) for i in line[:-6]])
                kernel = np.reshape(kernel, (120, 6))
                bias = np.array([float(i) for i in line[-6:]])
        
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
        with open(output_file, "a") as out_file:
            out_file.write("--- User " + user + "\n")
            out_file.write("Test Set Accuracy: {accuracy:0.3f}\nMean per Class Accuracy: {mean_perClass_accuracy:0.3f}\n".format(**eval_result))
            
    # Now we can calculate the mean accuracy between all users...
    cross_validation_accuracy = accuracy_sum / 9
    cross_validation_mean_per_class_accuracy = mean_class_accuracy_sum / 9
    # And write it on the output file.
    with open(output_file, "a") as out_file:
        out_file.write("\n\nFinal Cross Validation Accuracy: " + str(cross_validation_accuracy))
        out_file.write("\nFinal Cross Validation Mean Per Class Accuracy: " + str(cross_validation_mean_per_class_accuracy))
            
        
def perform_test02():
    # We need to create the dataset for the transfer learning model.
    model_dirs = {
        "a": "/home/buffellida/Tests/modelDir/dataset120-user-a",
        "b": "/home/buffellida/Tests/modelDir/dataset120-user-b",
        "c": "/home/buffellida/Tests/modelDir/dataset120-user-c",
        "d": "/home/buffellida/Tests/modelDir/dataset120-user-d",
        "e": "/home/buffellida/Tests/modelDir/dataset120-user-e",
        "f": "/home/buffellida/Tests/modelDir/dataset120-user-f",
        "g": "/home/buffellida/Tests/modelDir/dataset120-user-g",
        "h": "/home/buffellida/Tests/modelDir/dataset120-user-h",
        "i": "/home/buffellida/Tests/modelDir/dataset120-user-i"
        }
    print("Creating Dataset 120 ..")
    create_dataset(DATASET_120_PATH, "sepHARData_", model_dirs)
    print("Dataset 120 Created.")
    
    model_dirs = {
        "a": "/home/buffellida/Tests/modelDir/dataset12-user-a",
        "b": "/home/buffellida/Tests/modelDir/dataset12-user-b",
        "c": "/home/buffellida/Tests/modelDir/dataset12-user-c",
        "d": "/home/buffellida/Tests/modelDir/dataset12-user-d",
        "e": "/home/buffellida/Tests/modelDir/dataset12-user-e",
        "f": "/home/buffellida/Tests/modelDir/dataset12-user-f",
        "g": "/home/buffellida/Tests/modelDir/dataset12-user-g",
        "h": "/home/buffellida/Tests/modelDir/dataset12-user-h",
        "i": "/home/buffellida/Tests/modelDir/dataset12-user-i"
        }
    print("Creating Dataset 12 ..")
    create_dataset(DATASET_12_PATH, "user_", model_dirs)
    print("Dataset 12 Created.")
    
    
    # Create a directory where we will save all the model directories.
    model_dir_path = os.path.join(TESTS_DIR, "modelDir") 
    if not os.path.exists(model_dir_path):
        os.mkdir(model_dir_path)

# ---------------------------------- Dataset 120'000    ----------------------------------            
    with open(test02_results_filename, "a") as out_file:
        out_file.write("\n\n\n-------------------------- TEST 02 - Date: " + str(datetime.now()) + " --------------------------\n")
        out_file.write("------------- Dataset 120'000 -------------\n")
    print("------------- Dataset 120'000 -------------")
    cross_validation(DATASET_120_PATH, "sepHARData_", "dataset120", model_dir_path, test02_results_filename)
    
# ---------------------------------- Dataset 12'000    ----------------------------------    
    # Now same as before but for the 12'000 dataset
    with open(test02_results_filename, "a") as out_file:
        out_file.write("\n\n------------- Dataset 12'000 -------------\n")
    print("------------- Dataset 12'000 -------------")
    cross_validation(DATASET_12_PATH, "user_", "dataset12", model_dir_path, test02_results_filename)