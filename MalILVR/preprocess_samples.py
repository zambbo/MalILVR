from PIL import Image
import numpy as np
import os

SIZE=3514
DATASET_PATH="./generated_samples/"
TARGET_PATHS= ["benign_LogisticRegression_N4", "benign_RandomForest_N4", "benign_SupportVectorMachine_N4", "benign_DecisionTree_N4"]


def modifyNsave(target_path):
    SAMPLE_PATH = os.path.join(DATASET_PATH, target_path)
    SAVE_PATH = os.path.join(DATASET_PATH, target_path + "_npy")

    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    print(SAVE_PATH)
    for file in os.listdir(SAMPLE_PATH):
        if not file.endswith(".png"): continue
        print(file, end="\r")    
        save_path = os.path.join(SAVE_PATH, file[:-4] + ".npy")
        file_path = os.path.join(SAMPLE_PATH, file)
        
        image = Image.open(file_path).convert("L")
        
        array = np.array(image)
        
        array = array.flatten()
        array = array[:SIZE]
        array = array / 255
        array = array >= 0.5
        array = array.astype(np.int8)
        
        np.save(save_path, array)
        
def main():
    for target_path in TARGET_PATHS:
        modifyNsave(target_path)

if __name__ == "__main__":
    main()
