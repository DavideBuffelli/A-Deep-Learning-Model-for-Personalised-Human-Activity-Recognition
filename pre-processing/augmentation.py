import os
import csv
import numpy as np

"""
    Inside each user's folder (created by sepUsers.py) this script will create 
    (AUGMENTATION_NUMBER - 1) new accelerometer and gyroscope csv files which contain the 
    data of the orignal acceleromter and gyroscope csv files with added random noise. 
"""

USER_DATA_FOLDER_PATH = "/Path/To/Data/Dir"
AUGMENTATION_NUMBER = 10
ACC_NOISE_VAR = 0.5 # Variance of the random normal distribution for the noise added to the accelerometer data
GYRO_NOISE_VAR = 0.2 # Variance of the random normal distribution for the noise added to the gyroscope data

for user in ["a", "b", "c", "d", "e", "f", "g", "h", "i"]:
    print("Augmenting data for user ", user)
    for aug_num in range(1, AUGMENTATION_NUMBER):
        acc_file = os.path.join(USER_DATA_FOLDER_PATH, user, "accelerometer0.csv")
        gyro_file = os.path.join(USER_DATA_FOLDER_PATH, user, "gyroscope0.csv")
    
        acc_out_file = os.path.join(USER_DATA_FOLDER_PATH, user, "accelerometer" + str(aug_num) + ".csv")
        gyro_out_file = os.path.join(USER_DATA_FOLDER_PATH, user, "gyroscope" + str(aug_num) + ".csv")
        
        fieldnames = ["time", "x", "y", "z", "gt"]
        
        with open(acc_out_file, "w") as out_file:
            writeCSV = csv.DictWriter(out_file, fieldnames=fieldnames)
            
            with open(acc_file, "r") as in_file:
                readCSV = csv.DictReader(in_file, delimiter=",")
                for row in readCSV:
                    xzy_array = np.array([float(row["x"]), float(row["y"]), float(row["z"])])
                    xzy_array += np.random.normal(0.0, ACC_NOISE_VAR, size = (3,))
                    elem = {"time": row["time"], "x": xzy_array[0], "y": xzy_array[1], "z": xzy_array[2], "gt": row["gt"]}
                    writeCSV.writerow(elem)
                    
        
        with open(gyro_out_file, "w") as out_file:
            writeCSV = csv.DictWriter(out_file, fieldnames=fieldnames)
            
            with open(gyro_file, "r") as in_file:
                readCSV = csv.DictReader(in_file, delimiter=",")
                for row in readCSV:
                    xzy_array = np.array([float(row["x"]), float(row["y"]), float(row["z"])])
                    xzy_array += np.random.normal(0.0, GYRO_NOISE_VAR, size = (3,))
                    elem = {"time": row["time"], "x": xzy_array[0], "y": xzy_array[1], "z": xzy_array[2], "gt": row["gt"]}
                    writeCSV.writerow(elem)
                    
                    