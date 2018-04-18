import tensorflow as tf
import os

# ---------------------------- CONSTANTS ------------------------------------------------
# Directory Paths
TRAINING_DATA_FOLDER_PATH = "/Users/davidebuffelli/Desktop/Prova/Data/train"
EVAL_DATA_FOLDER_PATH = "/Users/davidebuffelli/Desktop/Prova/Data/eval"
MODEL_DIR_PATH = "/Users/davidebuffelli/Desktop/Prova/ModelDir"
SAVED_MODEL_DIR_PATH = "/Users/davidebuffelli/Desktop/Prova/SavedModelDir"

# Constants
BATCH_SIZE = 10 # Size of the batches of examples used for training and evaluation.
SAMPLE_LENGTH = 5.0 # Length in seconds of each sample that is contained in a .csv file.
TAO = 0.25 # Interval length
NUMBER_OF_INTERVALS = int(SAMPLE_LENGTH / TAO) # It has to be in integer.
MEASUREMENTS_PER_INTERVAL = 10
FEATURE_DIM = 2 * (3 * 2) * MEASUREMENTS_PER_INTERVAL # Detailed explanation in the thesis.
OUT_DIM = 6 # We have 6 different activities.

# ---------------------------- IMPORT DATA ----------------------------------------------
# Read a line of a .csv file and extract features and labels.
def read_csv(line):
	defaultVal = [[0.] for idx in range(NUMBER_OF_INTERVALS * FEATURE_DIM + OUT_DIM)] # default values in case of empty columns.
	fileData = tf.decode_csv(line, record_defaults=defaultVal) # Convert CSV records to tensors. Each column maps to one tensor.
	features = fileData[:NUMBER_OF_INTERVALS * FEATURE_DIM]
	features = tf.reshape(features, [NUMBER_OF_INTERVALS, FEATURE_DIM])
	labels = fileData[NUMBER_OF_INTERVALS * FEATURE_DIM:]
	labels = tf.reshape(labels, [OUT_DIM])
	
	# In the .csv files with the inputs we have that sometimes we don't have all the 20 intervals
	# but to make all the records of the same length they are padded with zeroes at the end.
	# We will then count the exact length of each sample and we will pass it as input, this
	# will then be useful for the RNN layers.
	used = tf.sign(tf.reduce_max(tf.abs(features), reduction_indices=1))
	real_sample_length = tf.reduce_sum(used, reduction_indices=0)
	real_sample_length = tf.cast(real_sample_length, tf.int32)
	
	return {"features":features, "length":real_sample_length}, labels

# Input function: creates an input pipeline that returns a dataset.
def input_fn(batch_size, training, data_folder_path):
	# Make a queue of file names including all the .csv files in the relative data directory.
	filename_queue = tf.train.match_filenames_once(os.path.join(data_folder_path, "*.csv"))
		
	dataset = tf.data.TextLineDataset(filename_queue) # Dataset that reads each file as a line of text.
	dataset = dataset.map(read_csv) # Trasform a line of text in a dictionary with features, length and labels.
	if training:
		dataset = dataset.shuffle(buffer_size=100)
	dataset = dataset.batch(BATCH_SIZE)
	
	return dataset

# Predict Input Function: input function used to predict the output of a single example.
def predict_input_fn(filename):
		dataset = tf.data.TextLineDataset(filename)
		dataset = dataset.map(read_csv)
		return dataset
# ---------------------------- CREATE MODEL ---------------------------------------------
# Model function: creates the DeepSense estimator.
def deepSense_model_fn(features, labels, mode, params):
	f = features["features"]
	length = features["length"]
	
	# When we are in TRAIN or in EVAL mode, features has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, FEATURE_DIM)
	# but, when we are in PREDICT mode features corresponds to a single element, so it has shape
	# (NUMBER_OF_INTERVALS, FEATURE_DIM). 
	# TensorFlow methods require an input of shape (BATCH_SIZE, NUMBER_OF_INTERVALS, FEATURE_DIM, CHANNELS)
	# se we have to do some reshaping.
	if mode == tf.estimator.ModeKeys.PREDICT:
		f = tf.reshape(f, [1, NUMBER_OF_INTERVALS, FEATURE_DIM])
		length = tf.reshape(length, [1]) # Make it a tensor instead of a scalar.
	sensor_inputs = tf.expand_dims(f, axis=-1) # Add dimension for the channel.
	
	# Obtain the batch size, it will be necessary later for the RNN part.
	batch_size = tf.shape(f)[0]
	
	# Separate accelerometer data from gyroscope data.
	acc_inputs, gyro_inputs = tf.split(sensor_inputs, num_or_size_splits=2, axis=2)
	
	#------ Accelerometer Individual Convolutional Layers
	acc_conv1 = tf.layers.conv2d(acc_inputs, params["acc_conv1_num_filters"], params["acc_conv1_kernel_size"],
		params["acc_conv1_stride"], params["acc_conv1_padding"])
	acc_conv1 = tf.layers.batch_normalization(acc_conv1, training=(mode == tf.estimator.ModeKeys.TRAIN))
	acc_conv1 = tf.nn.relu(acc_conv1)
	acc_conv1_shape = acc_conv1.get_shape().as_list() # Use this to define the shape of the dropout mask.
	acc_conv1 = tf.layers.dropout(acc_conv1, rate=params["acc_conv1_dropout_rate"],
		noise_shape=[acc_conv1_shape[0], 1, 1, acc_conv1_shape[3]], training=(mode == tf.estimator.ModeKeys.TRAIN))

	acc_conv2 = tf.layers.conv2d(acc_conv1, params["acc_conv2_num_filters"], params["acc_conv2_kernel_size"],
		params["acc_conv2_stride"], params["acc_conv2_padding"])
	acc_conv2 = tf.layers.batch_normalization(acc_conv2, training=(mode == tf.estimator.ModeKeys.TRAIN))
	acc_conv2 = tf.nn.relu(acc_conv2)
	acc_conv2_shape = acc_conv2.get_shape().as_list()
	acc_conv2 = tf.layers.dropout(acc_conv2, rate=params["acc_conv2_dropout_rate"],
		noise_shape=[acc_conv2_shape[0], 1, 1, acc_conv2_shape[3]], training=(mode == tf.estimator.ModeKeys.TRAIN))

	acc_conv3 = tf.layers.conv2d(acc_conv2, params["acc_conv3_num_filters"], params["acc_conv3_kernel_size"],
		params["acc_conv3_stride"], params["acc_conv3_padding"])
	acc_conv3 = tf.layers.batch_normalization(acc_conv3, training=(mode == tf.estimator.ModeKeys.TRAIN))
	acc_conv3 = tf.nn.relu(acc_conv3)
	
	# Reshape for future concatenation with gyroscope conv_out
	acc_conv3_shape = acc_conv3.get_shape().as_list()
	acc_conv_out = tf.reshape(acc_conv3, [-1, acc_conv3_shape[1], 1, acc_conv3_shape[2], acc_conv3_shape[3]])
	
	#------ Gyroscope Individual Convolutional Layers
	gyro_conv1 = tf.layers.conv2d(gyro_inputs, params["gyro_conv1_num_filters"], params["gyro_conv1_kernel_size"],
		params["gyro_conv1_stride"], params["gyro_conv1_padding"])
	gyro_conv1 = tf.layers.batch_normalization(gyro_conv1, training=(mode == tf.estimator.ModeKeys.TRAIN))
	gyro_conv1 = tf.nn.relu(gyro_conv1)
	gyro_conv1_shape = gyro_conv1.get_shape().as_list()
	gyro_conv1 = tf.layers.dropout(gyro_conv1, rate=params["gyro_conv1_dropout_rate"],
		noise_shape=[gyro_conv1_shape[0], 1, 1, gyro_conv1_shape[3]], training=(mode == tf.estimator.ModeKeys.TRAIN))
		
	gyro_conv2 = tf.layers.conv2d(gyro_conv1, params["gyro_conv2_num_filters"], params["gyro_conv2_kernel_size"],
		params["gyro_conv2_stride"], params["gyro_conv2_padding"])
	gyro_conv2 = tf.layers.batch_normalization(gyro_conv2, training=(mode == tf.estimator.ModeKeys.TRAIN))
	gyro_conv2 = tf.nn.relu(gyro_conv2)
	gyro_conv2_shape = gyro_conv2.get_shape().as_list()
	gyro_conv2 = tf.layers.dropout(gyro_conv2, rate=params["gyro_conv2_dropout_rate"],
		noise_shape=[gyro_conv2_shape[0], 1, 1, gyro_conv2_shape[3]], training=(mode == tf.estimator.ModeKeys.TRAIN))
		
	gyro_conv3 = tf.layers.conv2d(gyro_conv2, params["gyro_conv3_num_filters"], params["gyro_conv3_kernel_size"],
		params["gyro_conv3_stride"], params["gyro_conv3_padding"])
	gyro_conv3 = tf.layers.batch_normalization(gyro_conv3, training=(mode == tf.estimator.ModeKeys.TRAIN))
	gyro_conv3 = tf.nn.relu(gyro_conv3)
	
	gyro_conv3_shape = gyro_conv3.get_shape().as_list()
	gyro_conv_out = tf.reshape(gyro_conv3, [-1, gyro_conv3_shape[1], 1, gyro_conv3_shape[2], gyro_conv3_shape[3]])
	
	# Concatenate the output of the individual convolutional layers for the two sensors and then apply dropout.
	sensor_conv_in = tf.concat([acc_conv_out, gyro_conv_out], 2)
	# sensor_conv_in has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, 2, ...(depends on kernel_size and padding of conv layers), CHANNELS(=number of filters of last conv layer)).
	senor_conv_shape = sensor_conv_in.get_shape().as_list()
	sensor_conv_in = tf.layers.dropout(sensor_conv_in, params["sensor_conv_in_dropout_rate"],
		noise_shape=[senor_conv_shape[0], 1, 1, 1, senor_conv_shape[4]], training=(mode == tf.estimator.ModeKeys.TRAIN))
	
	#------ Merge Convolutional Layers	
	sensor_conv1 = tf.layers.conv3d(sensor_conv_in, params["sensor_conv1_num_filters"], params["sensor_conv1_kernel_size"],
		params["sensor_conv1_stride"], params["sensor_conv1_padding"])
	sensor_conv1 = tf.layers.batch_normalization(sensor_conv1, training=(mode == tf.estimator.ModeKeys.TRAIN))
	sensor_conv1 = tf.nn.relu(sensor_conv1)
	sensor_conv1_shape = sensor_conv1.get_shape().as_list()
	sensor_conv1 = tf.layers.dropout(sensor_conv1, rate=params["sensor_conv1_dropout_rate"],
		noise_shape=[sensor_conv1_shape[0], 1, 1, 1, sensor_conv1_shape[4]], training=(mode == tf.estimator.ModeKeys.TRAIN))
		
	sensor_conv2 = tf.layers.conv3d(sensor_conv1, params["sensor_conv2_num_filters"], params["sensor_conv2_kernel_size"],
		params["sensor_conv2_stride"], params["sensor_conv2_padding"])
	sensor_conv2 = tf.layers.batch_normalization(sensor_conv2, training=(mode == tf.estimator.ModeKeys.TRAIN))
	sensor_conv2 = tf.nn.relu(sensor_conv2)
	sensor_conv2_shape = sensor_conv2.get_shape().as_list()
	sensor_conv2 = tf.layers.dropout(sensor_conv2, rate=params["sensor_conv2_dropout_rate"],
		noise_shape=[sensor_conv2_shape[0], 1, 1, 1, sensor_conv2_shape[4]], training=(mode == tf.estimator.ModeKeys.TRAIN))
		
	sensor_conv3 = tf.layers.conv3d(sensor_conv2, params["sensor_conv3_num_filters"], params["sensor_conv3_kernel_size"],
		params["sensor_conv3_stride"], params["sensor_conv3_padding"])
	sensor_conv3 = tf.layers.batch_normalization(sensor_conv3, training=(mode == tf.estimator.ModeKeys.TRAIN))
	sensor_conv3 = tf.nn.relu(sensor_conv3)

	# Reshape for Recurrent Neural Network.
	sensor_conv3_shape = sensor_conv3.get_shape().as_list()
	sensor_conv_out = tf.reshape(sensor_conv3, [-1, sensor_conv3_shape[1], sensor_conv3_shape[2]*sensor_conv3_shape[3]*sensor_conv3_shape[4]])
	# sensor_conv_out has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, ...(depends on kernel_size and padding of sensor conv layers))
	
	#------ RNN Layers
	# We need to use the contrib module because RNN functions are still "experimental".
	gru_cell1 = tf.contrib.rnn.GRUCell(FEATURE_DIM)
	if mode == tf.estimator.ModeKeys.TRAIN:
		gru_cell1 = tf.contrib.rnn.DropoutWrapper(gru_cell1, output_keep_prob=params["gru_cell1_dropout_output_keep_prob"])

	gru_cell2 = tf.contrib.rnn.GRUCell(FEATURE_DIM)
	if mode == tf.estimator.ModeKeys.TRAIN:
		gru_cell2 = tf.contrib.rnn.DropoutWrapper(gru_cell2, output_keep_prob=params["gru_cell2_dropout_output_keep_prob"])

	cell = tf.contrib.rnn.MultiRNNCell([gru_cell1, gru_cell2])
	init_state = cell.zero_state(batch_size, tf.float32)

	cell_output, final_stateTuple = tf.nn.dynamic_rnn(cell, sensor_conv_out, sequence_length=length, initial_state=init_state, time_major=False)
	# cell_output has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, cell_output_dim(in our case it is = FEATURE_DIM)).

	# Sum the output of the RNN for each example and calculate the mean.
	sum_cell_out = tf.reduce_sum(cell_output, axis=1, keepdims=False)
	l = tf.reshape(length, [batch_size, 1]) 
	l = tf.cast(l, tf.float32)
	avg_cell_out = sum_cell_out/(tf.tile(l, [1, FEATURE_DIM])) # we have to calculate the mean this way to take into account for the different lengths.

	#------ Output Layer
	# Final Fully Connected Layer with one output unit per possible class.
	logits = tf.layers.dense(avg_cell_out, OUT_DIM)

	predicted_classes = tf.argmax(logits, 1)
	
	# PREDICTION MODE
	if mode == tf.estimator.ModeKeys.PREDICT:
		# Create a dictionary of predictions to be passed as output.
		predictions = {
		"class_ids": predicted_classes[:, tf.newaxis],
		"probabilities": tf.nn.softmax(logits),
		"logits": logits,
		"tl_input": avg_cell_out # This is needed only for transfer learning(it becomes the input of the user-specific output layer).
		}
		# With the export_outputs we define the outputs available for the SavedModel.
		return tf.estimator.EstimatorSpec(mode, predictions=predictions, 
			export_outputs={"classify:":tf.estimator.export.PredictOutput(predictions)})

	# Calculate the loss(Cross Entropy).
	batchLoss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
	loss = tf.reduce_mean(batchLoss)
        
	# EVALUATION MODE
	if mode == tf.estimator.ModeKeys.EVAL:
		labels = tf.argmax(labels, 1)
		# Add some metrics that will be calculated during evaluation.
		accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name="accuracy_op")
		mean_perClass_accuracy = tf.metrics.mean_per_class_accuracy(labels, predicted_classes, OUT_DIM, name="mean_perClass_accuracy_op")
		metrics = {"accuracy": accuracy, "mean_perClass_accuracy": mean_perClass_accuracy}
		tf.summary.scalar("Accuracy", accuracy[1]) # for TensorBoard.
		tf.summary.scalar("Mean Per Class Accuracy", mean_perClass_accuracy[1]) # for TensorBoard.
		return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
	
	# TRAINING MODE
	if mode == tf.estimator.ModeKeys.TRAIN:
		# For training I used the same optimizer used by the authors of the DeepSense paper.
		optimizer = tf.train.AdamOptimizer(
			learning_rate=1e-5,
			beta1=0.5,
			beta2=0.9)
		train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
		return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

# ----------------------------- USAGE EXAMPLE -------------------------------------------
if __name__ == "__main__":
	# -------------- CREATE ESTIMATOR
	# Wrap DeepSense estimator in a tf.estimator.Estimator, passing all parameters.
	deepSense_classifier = tf.estimator.Estimator(
		model_fn = deepSense_model_fn,
		model_dir = MODEL_DIR_PATH,
		params = {
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
			"gru_cell2_dropout_output_keep_prob": 0.5
		})

	# -------------- TRAIN ESTIMATOR
	# tf.estimator.Estimator wants an input function with no arguments, so wrap input_fn in a lambda.
	deepSense_classifier.train(lambda:input_fn(BATCH_SIZE, True, TRAINING_DATA_FOLDER_PATH), steps=1)

	# -------------- EVALUATE METRICS
	eval_result = deepSense_classifier.evaluate(lambda:input_fn(BATCH_SIZE, False, EVAL_DATA_FOLDER_PATH), steps=1)
	print("\nTest Set Accuracy: {accuracy:0.3f}\nMean per Class Accuracy: {mean_perClass_accuracy:0.3f}".format(**eval_result))

	# -------------- SAVE MODEL IN SAVED_MODEL FORMAT
	# TensorFlow Estimators already save all the information about the model using checkpoints(saved in the MODEL_DIR_PATH)
	# but if we want to save this information in a format independent of the code that created the model
	# we need to use the SavedModel format(saved in SAVED_MODEL_DIR_PATH). Models saved this way
	# can then be used with TensorFlow Serving and other tools. 
	# Estimators already have a method to export SavedModels, you only need to pass a serving_input_function
	# with the placeholders for the inputs.
	# An example of usage of the SavedModel is in the file "savedModelTest.py".
	deepSense_classifier.export_savedmodel(
		SAVED_MODEL_DIR_PATH,
		tf.estimator.export.build_raw_serving_input_receiver_fn({"features":tf.placeholder(tf.float32, shape=[NUMBER_OF_INTERVALS, FEATURE_DIM]), "length":tf.placeholder(tf.int64, shape=[1])}))

	# -------------- EXAMPLE OF PREDICTION
	predictions = deepSense_classifier.predict(lambda:predict_input_fn("/Users/davidebuffelli/Desktop/Prova/Data/eval/eval_18.csv"))

	for p in predictions:
		print("Predicted Class: ", p["class_ids"])
		print("Probabilities: ", p["probabilities"])
		print("Logits: ", p["logits"])
		print("Transfer Learning input: ", p["tl_input"])
		
	kernel = deepSense_classifier.get_variable_value("dense/kernel")
	bias = deepSense_classifier.get_variable_value("dense/bias")
	print(kernel.shape)
	print(bias.shape)
	"""
	for transfer learning
	print(deepSense_classifier.get_variable_names())
	kernel = deepSense_classifier.get_variable_value("dense/kernel")
	bias = deepSense_classifier.get_variable_value("dense/bias")
	print(kernel.shape)
	print(bias.shape)
	"""