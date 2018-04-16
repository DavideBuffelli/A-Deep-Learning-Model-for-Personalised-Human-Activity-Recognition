import tensorflow as tf
import os
import numpy as np

MODEL_DIR_PATH = ""
FEATURES_DIM = 120
OUT_DIM = 6

# ---------------------------- IMPORT DATA ----------------------------------------------
# Read a line of a .csv file and extract features and labels.
def read_csv(line):
	defaultVal = [[0.] for idx in range(FEATURES_DIM + OUT_DIM)] # default values in case of empty columns.
	fileData = tf.decode_csv(line, record_defaults=defaultVal) # Convert CSV records to tensors. Each column maps to one tensor.
	features = fileData[:-OUT_DIM]
	features = tf.reshape(features, [FEATURES_DIM])
	label = fileData[-OUT_DIM:]
	label = tf.reshape(label, [OUT_DIM])
	
	return features, label

# Input function: creates an input pipeline that returns a dataset.
def tl_input_fn(data_folder_path):
	# Make a queue of file names including all the .csv files in the relative data directory.
	filename_queue = tf.train.match_filenames_once(os.path.join(data_folder_path, "*.csv"))
		
	dataset = tf.data.TextLineDataset(filename_queue) # Dataset that reads each file as a line of text.
	dataset = dataset.map(read_csv) # Trasform a line of text in a dictionary with features, length and labels.
	
	return dataset.batch(1)

# Predict input function: creates a dataset witha single example.
def tl_predict_input_fn(filename):
		dataset = tf.data.TextLineDataset(filename)
		dataset = dataset.map(read_csv)
		return dataset.batch(1)
	
# ---------------------------- CREATE MODEL ---------------------------------------------	
# Transfer learning using the last layer of DeepSense(you have to modify DeepSense so that
# it gives avg_cell_out as output, which then become the input of this model). So we are only
# training the last layer to a specific user, not the entire DeepSense classifier.
def tl_model_fn(features, labels, mode, params):
	# features has shape (batch_size, 120). labels has shape (batch_size, OUT_DIM)

	logits = tf.layers.dense(
		inputs = features, 
		units = OUT_DIM,
		kernel_initializer = tf.constant_initializer(params["kernel_value"]),
		bias_initializer = tf.constant_initializer(params["bias_value"]))
		# kernel_value and bias_value are taken from the trained DeepSense classifier

	predicted_classes = tf.argmax(logits, 1)
	
	# PREDICTION MODE
	if mode == tf.estimator.ModeKeys.PREDICT:
		# Create a dictionary of predictions to be passed as output.
		predictions = {
		'class_ids': predicted_classes[:, tf.newaxis],
		'probabilities': tf.nn.softmax(logits),
		'logits': logits
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
		accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='accuracy_op')
		mean_perClass_accuracy = tf.metrics.mean_per_class_accuracy(labels, predicted_classes, OUT_DIM, name="mean_perClass_accuracy_op")
		metrics = {'accuracy': accuracy, 'mean_perClass_accuracy': mean_perClass_accuracy}
		tf.summary.scalar('Accuracy', accuracy[1]) # for TensorBoard.
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
	# Create the transfer learning estimator.
	# For this example I am using random kernel and bias, but in a real case you want to use
	# the kernel and bias obtained from the output layer of a trained DeepSense model.
	deepSense_transferLearning_classifier = tf.estimator.Estimator(
		model_fn = tl_model_fn,
		model_dir = "/Users/davidebuffelli/Desktop/Prova/ModelDirTL",
		params = {"kernel_value": np.random.rand(FEATURES_DIM, OUT_DIM), 
					"bias_value": np.random.rand(OUT_DIM)})
	
	# Train.
	deepSense_transferLearning_classifier.train(lambda:tl_input_fn("/Users/davidebuffelli/Desktop/Data/tl_sepHARData_a/train"))
	
	# Predict.
	predictions = deepSense_transferLearning_classifier.predict(lambda:tl_predict_input_fn("/Users/davidebuffelli/Desktop/Data/tl_sepHARData_a/eval/eval_200.csv"))
	pred = next(predictions)
	print(pred['class_ids'])