import pickle
import os
import numpy as np

blackbox_model_names = ["DecisionTree", "LogisticRegression", "RandomForest", "SupportVectorMachine"]
BLACKBOX_BASE_PATH = "../blackbox"
DATASET_PATH = "./datasets/"
BENIGN_PATH = os.path.join(DATASET_PATH, "benigns")
MALWARE_PATH = os.path.join(DATASET_PATH, "malwares")

SIZE=3514

BENIGN=0

def get_data(path: str) -> list[list]:
    ret_arr = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        print(file_path, end="\r")

        array = np.load(file_path)
        array = array.flatten()
        array = array[:SIZE]
        array = list(array)
        ret_arr.append(array)

    return ret_arr

def main():
    data = get_data(BENIGN_PATH)
    data.extend(get_data(MALWARE_PATH))
    print()
    for model_name in blackbox_model_names:
        path = os.path.join(BLACKBOX_BASE_PATH, model_name + '.pickle')
        save_path = os.path.join(DATASET_PATH, "benign_" + model_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        model = None
        print(model_name)
        with open(path, "rb") as f:
            model = pickle.load(f)

            y = model.predict(data)
            cnt = 0
            for (i, y_cell) in enumerate(y):
                
                if y_cell == BENIGN:
                    array = data[i]
                    array = np.array(array)
                    array = np.pad(array, (0,64*64-len(array)), constant_values=0)
                    array = array.reshape(64, 64, 1)
                    final_save_path = f"{save_path}/{cnt}.npy"
                    np.save(final_save_path, array)
                    print(final_save_path, end="\r")
                    cnt += 1
            
if __name__ == "__main__":
    main()
