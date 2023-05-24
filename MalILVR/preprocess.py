import numpy as np
import os

SIZE=3514
DATASET_PATH="./datasets/"
BENIGN_PATH=os.path.join(DATASET_PATH, "benigns")
MALWARE_PATH=os.path.join(DATASET_PATH, "malwares")

for path in [BENIGN_PATH, MALWARE_PATH]:
    save_base_path = path + "_1d"
    for file in os.listdir(path):
        print(file, end="\r")
        file_path = os.path.join(path, file)

        array = np.load(file_path)
        
        array = array.flatten()
        array = array[:SIZE]
        
        save_file_path = os.path.join(save_base_path, file[:-4] + '_1d.npy')
        np.save(save_file_path, array)
        



