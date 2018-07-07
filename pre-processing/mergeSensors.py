import os
import csv
import numpy as np
from scipy.interpolate import interp1d
from scipy.fftpack import fft

"""
	Run after running sepUsers.py. This will "merge" the measurements
	from accelerometer and gyroscope, doing all the necessary pre-processing(as described in
	the thesis).  
"""

user_folders_Dir = "/Path/To/Data/Dir"
users_names = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
activities = ["bike", "sit", "stand", "walk", "stairsup", "stairsdown"]
AUGMENTATION_NUMBER = 10

SAMPLE_LENGTH = 5.0
INTERVAL_LENGTH = 0.25 #TAO
NUMBER_OF_INTERVALS = int(SAMPLE_LENGTH / INTERVAL_LENGTH)
MEASUREMENTS_PER_INTERVAL = 10
TOT_FEATURES_PER_SAMPLE = 2 * 6 * MEASUREMENTS_PER_INTERVAL * NUMBER_OF_INTERVALS 

def interp_and_FFT(interval_times, interval_values):
	# Must have at least to measurements to be able to interpolate.
	if len(interval_times) < 2:
		# In this case we just return the FFT of the only measurement we have and we pad with zeroes.
		out = [fft(interval_values[0])]
		out.extend([np.array([np.complex(0.0, 0.0), np.complex(0.0, 0.0), np.complex(0.0, 0.0)]) for _ in range(MEASUREMENTS_PER_INTERVAL-1)])
		return out
		
	interpolation_function = interp1d(interval_times, interval_values, axis = 0)
	points = np.linspace(interval_times[0], interval_times[-1], MEASUREMENTS_PER_INTERVAL)
	interp_values = interpolation_function(points)
	FFT_values = fft(interp_values)
	return FFT_values # returns a list of complex-valued arrays.

file_index = 0
def merge_mesaurements(acc_measurements, gyro_measurements, output_dir, aug_num):
	# In some cases there is too little data, and there are some measurements that have
	# only data for the accelerometer. We discard these cases.
	if len(acc_measurements) < 30 or len(gyro_measurements) == 0:
		return
		
	output = [] # This will contain all the values to be written in output.
	
	acc_index = 0	
	gyro_index = 0
	while acc_index < len(acc_measurements):
		# Find start and end index in the accelerometer data of the current interval of INTERVAL_LENGTH.
		acc_starting_index = acc_index
		last_timestamp = int(acc_measurements[acc_index]["time"]) / 1000000000
		time_count = 0.0
		while time_count + (int(acc_measurements[acc_index]["time"]) / 1000000000) - last_timestamp < INTERVAL_LENGTH:
			time_count += (int(acc_measurements[acc_index]["time"]) / 1000000000) - last_timestamp
			last_timestamp = int(acc_measurements[acc_index]["time"]) / 1000000000
			acc_index +=1
			if acc_index >= len(acc_measurements):
				break
		
		# Find start and end indexes for the gyroscope.
		gyro_starting_index = gyro_index
		while gyro_index < len(gyro_measurements) and (int(gyro_measurements[gyro_index]["time"]) / 1000000000) <= (int(acc_measurements[acc_index-1]["time"]) / 1000000000):
			gyro_index += 1
			
		# Group the measurements in this interval in a list of numpy arrays(with the values for the axes).
		acc_interval = []
		acc_interval_times = []
		for i in range(acc_starting_index, acc_index):
			measurement = acc_measurements[i]
			x = measurement["x"]
			y = measurement["y"]
			z = measurement["z"]
			acc_interval.append(np.array([x, y, z]))
			acc_interval_times.append((int(measurement["time"]) / 1000000000))
		gyro_interval = []
		gyro_interval_times = []
		for i in range(gyro_starting_index, gyro_index):
			measurement = gyro_measurements[i]
			x = measurement["x"]
			y = measurement["y"]
			z = measurement["z"]
			gyro_interval.append(np.array([x, y, z]))
			gyro_interval_times.append((int(measurement["time"]) / 1000000000))
		
		# Interpolate the values in the interval(to deal with the fact that measurements 
		# are not evenly "spaced" in time). Then take 10 evenly spaced
		# points, apply FFT and save the results in the output.
		acc_FFT = interp_and_FFT(acc_interval_times, acc_interval)
		for array in acc_FFT:
			for elem in array:
				output.append(elem.real)
				output.append(elem.imag)
		# Same for gyroscope.
		if len(gyro_interval) == 0: # sometimes we have only accelerometer measurements for an interval(the viceversa can never happen, because of how we collect intervals)
			continue
		gyro_FFT = interp_and_FFT(gyro_interval_times, gyro_interval)
		for array in gyro_FFT:
			for elem in array:
				output.append(elem.real)
				output.append(elem.imag)
	
	# Previous loop finishes when accelerometer measurements finish, but we might be in a case 
	# where the sample was shorter than SAMPLE_LENGTH. We simply pad it with zeroes at the end.
	# (We will use this information in the DeepSense framework).		
	if len(output) < TOT_FEATURES_PER_SAMPLE:
		output.extend([0.0 for _ in range(TOT_FEATURES_PER_SAMPLE-len(output))])
	
	# Add ground truth at the end of output.
	gt = acc_measurements[0]["gt"]
	gt_encoding = {"bike":[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
					"sit":[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 
					"stand":[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 
					"walk":[0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 
					"stairsup":[0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 
					"stairsdown":[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}[gt]
	output.extend(gt_encoding)

	# Write output.
	global file_index
	gt_folder = os.path.join(output_dir, gt)
	with open(os.path.join(gt_folder, str(file_index) + "_" + str(aug_num) +".csv"), "w") as writeFile:
		writer = csv.writer(writeFile)
		writer.writerow(output)
		file_index += 1



for user in users_names:
	print("Current User: ", user)
	user_dir = os.path.join(user_folders_Dir, user)
	for aug_num in range(AUGMENTATION_NUMBER):
		acc_file = os.path.join(user_dir, "accelerometer" + str(aug_num) + ".csv")
		gyro_file = os.path.join(user_dir, "gyroscope" + str(aug_num) + ".csv")

		# Create a directory named final where we will have all the activities folder where we will
		# put all the output files(one .csv file per SAMPLE_LENGTH seconds sample).
		output_dir = os.path.join(user_dir, "final")
		if not os.path.exists(output_dir):
			os.mkdir(output_dir)
		for activity in activities:
			if not os.path.exists(os.path.join(output_dir, activity)):
				os.mkdir(os.path.join(output_dir, activity))

		with open(acc_file, "r") as accFile, open(gyro_file, "r") as gyroFile:
			readAcc = csv.DictReader(accFile, delimiter=",")
			readGyro = csv.DictReader(gyroFile, delimiter=",")
		
			# Different devices use different starting points for their timestamps(we can use
			# this to distinguish devices), but they are always in nanoseconds, so dividing by
			# 1000000000 we get the value in seconds.
			acc_row = next(readAcc, None)
			while acc_row is not None:
				# Get all the accelerometer measurements(of the same device and of the same activity)
				# that compose a SAMPLE_LENGTH seconds interval.
				if acc_row is not None:
					current_gt = acc_row["gt"]
					current_timeStamp_length = len(acc_row["time"])
					start_time = int(acc_row["time"]) / 1000000000
					last_timestamp = start_time
					time_count = 0.0
					interval_acc_measurements = []
					while time_count + (int(acc_row["time"]) / 1000000000) - last_timestamp < SAMPLE_LENGTH and acc_row["gt"] == current_gt and len(acc_row["time"]) == current_timeStamp_length:
						interval_acc_measurements.append(acc_row)
						row_timestamp = int(acc_row["time"]) / 1000000000
						time_count += row_timestamp - last_timestamp
						last_timestamp = row_timestamp
						acc_row = next(readAcc, None)
						if acc_row is None:
							break
			
					# Get the gyroscope samples that were taken in the same time as the accelerometer ones.
					gyro_row = next(readGyro, None)
					while gyro_row is not None and (int(gyro_row["time"]) / 1000000000) < start_time and abs((int(gyro_row["time"]) / 1000000000) - start_time) > 0.01 \
							and gyro_row["gt"] == current_gt and len(gyro_row["time"]) == current_timeStamp_length:
						gyro_row = next(readGyro, None)
					interval_gyro_measurements = []
					while gyro_row is not None and (int(gyro_row["time"]) / 1000000000) < last_timestamp \
							and gyro_row["gt"] == current_gt and len(gyro_row["time"]) == current_timeStamp_length:
						interval_gyro_measurements.append(gyro_row)
						gyro_row = next(readGyro, None)
				
					# Pass accelerometer and gyroscope samples to merge_mesaurements to create 
					# the final output file for this SAMPLE_LENGTH seconds sample.
					merge_mesaurements(interval_acc_measurements, interval_gyro_measurements, output_dir, aug_num)