import os
import numpy as np

BASE_DIR="./generated_samples"

def preprocess_array(array: np.array) -> np.array:
    array = array > 0.5
    array = array.astype(np.int8)
    return array

def preprocess():
    
    for file in os.listdir(BASE_DIR):
        save_dir_name = file.split(".")[0]
        save_dir_name = os.path.join(BASE_DIR, save_dir_name)

        if not os.path.exists(save_dir_name):
            os.mkdir(save_dir_name)

        array = np.load(os.path.join(BASE_DIR, file))
        print(save_dir_name)

        for i, arr in enumerate(array):
            print(i, end="\r")
            arr = preprocess_array(arr)

            save_file_path = os.path.join(save_dir_name, str(i))
            np.save(save_file_path, arr)

        print()

def main():
    preprocess()


if __name__ == "__main__":
    main()
    
