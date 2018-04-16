import sys
import os
import csv

dataDir = "/Users/davidebuffelli/Desktop/Data/Activityrecognitionexp"
exportDir = "/Users/davidebuffelli/Desktop/Data"

# Create one folder for each user, and inside each folder create two .csv files that will
# contain accelerometer and gyroscope data for that user.
users_names = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
fieldnames = ["time", "x", "y", "z", "gt"] # fields that we will have in the output .csv files.
for user in users_names:
	user_folder_path = os.path.join(exportDir, user)
	if not os.path.exists(user_folder_path):
		os.mkdir(user_folder_path)
		with open(os.path.join(user_folder_path, "accelerometer.csv"), mode='w') as csvFile:
			writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
			writer.writeheader()
		with open(os.path.join(user_folder_path, "gyroscope.csv"), mode='w') as csvFile:
			writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
			writer.writeheader()

# Extract accelerometer data foe each user and save it in the accelerometer.csv file
# in his folder. We are interested only in the fields: "Creation_Time", "x", "y", "z" and "gt".
file_name = [os.path.join(dataDir, "Phones_accelerometer.csv")]
with open(os.path.join(exportDir, "a", "accelerometer.csv"), 'a') as curWriteFileA,\
	open(os.path.join(exportDir, "b", "accelerometer.csv"), 'a') as curWriteFileB,\
	open(os.path.join(exportDir, "c", "accelerometer.csv"), 'a') as curWriteFileC,\
	open(os.path.join(exportDir, "d", "accelerometer.csv"), 'a') as curWriteFileD,\
	open(os.path.join(exportDir, "e", "accelerometer.csv"), 'a') as curWriteFileE,\
	open(os.path.join(exportDir, "f", "accelerometer.csv"), 'a') as curWriteFileF,\
	open(os.path.join(exportDir, "g", "accelerometer.csv"), 'a') as curWriteFileG,\
	open(os.path.join(exportDir, "h", "accelerometer.csv"), 'a') as curWriteFileH,\
	open(os.path.join(exportDir, "i", "accelerometer.csv"), 'a') as curWriteFileI,\
	open(file_name) as csvInputFile:
	
	writers = {"a":csv.DictWriter(curWriteFileA, fieldnames=fieldnames),
			"b":csv.DictWriter(curWriteFileB, fieldnames=fieldnames),
			"c":csv.DictWriter(curWriteFileC, fieldnames=fieldnames),
			"d":csv.DictWriter(curWriteFileD, fieldnames=fieldnames),
			"e":csv.DictWriter(curWriteFileE, fieldnames=fieldnames),
			"f":csv.DictWriter(curWriteFileF, fieldnames=fieldnames),
			"g":csv.DictWriter(curWriteFileG, fieldnames=fieldnames),
			"h":csv.DictWriter(curWriteFileH, fieldnames=fieldnames),
			"i":csv.DictWriter(curWriteFileI, fieldnames=fieldnames)}

	readCSV = csv.DictReader(csvInputFile, delimiter=',')
	for row in readCSV:
		if row["gt"] != "null":
			elem = {"time": row["Creation_Time"], "x":row["x"], "y":row["y"], "z":row["z"], "gt":row["gt"]}
			writers[row["User"]].writerow(elem)
				
	
# Same as before but for gyroscope data.			
file_name = [os.path.join(dataDir, "Phones_gyroscope.csv")]
with open(os.path.join(exportDir, "a", "gyroscope.csv"), 'a') as curWriteFileA,\
	open(os.path.join(exportDir, "b", "gyroscope.csv"), 'a') as curWriteFileB,\
	open(os.path.join(exportDir, "c", "gyroscope.csv"), 'a') as curWriteFileC,\
	open(os.path.join(exportDir, "d", "gyroscope.csv"), 'a') as curWriteFileD,\
	open(os.path.join(exportDir, "e", "gyroscope.csv"), 'a') as curWriteFileE,\
	open(os.path.join(exportDir, "f", "gyroscope.csv"), 'a') as curWriteFileF,\
	open(os.path.join(exportDir, "g", "gyroscope.csv"), 'a') as curWriteFileG,\
	open(os.path.join(exportDir, "h", "gyroscope.csv"), 'a') as curWriteFileH,\
	open(os.path.join(exportDir, "i", "gyroscope.csv"), 'a') as curWriteFileI,\
	open(file_name) as csvInputFile:
	
	writers = {"a":csv.DictWriter(curWriteFileA, fieldnames=fieldnames),
			"b":csv.DictWriter(curWriteFileB, fieldnames=fieldnames),
			"c":csv.DictWriter(curWriteFileC, fieldnames=fieldnames),
			"d":csv.DictWriter(curWriteFileD, fieldnames=fieldnames),
			"e":csv.DictWriter(curWriteFileE, fieldnames=fieldnames),
			"f":csv.DictWriter(curWriteFileF, fieldnames=fieldnames),
			"g":csv.DictWriter(curWriteFileG, fieldnames=fieldnames),
			"h":csv.DictWriter(curWriteFileH, fieldnames=fieldnames),
			"i":csv.DictWriter(curWriteFileI, fieldnames=fieldnames)}

	readCSV = csv.DictReader(csvInputFile, delimiter=',')
	for row in readCSV:
		if row["gt"] != "null":
			elem = {"time": row["Creation_Time"], "x":row["x"], "y":row["y"], "z":row["z"], "gt":row["gt"]}
			writers[row["User"]].writerow(elem)