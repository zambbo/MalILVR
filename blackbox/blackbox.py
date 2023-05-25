from sklearn import tree
import numpy as np
import os
import random
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

DATASET_PATH = "../datasets/"
BENIGN_PATH = os.path.join(DATASET_PATH, "benigns")
MALWARE_PATH = os.path.join(DATASET_PATH, "malwares")

SIZE=3514

models = ["DecisionTree", "LogisticRegression", "RandomForest", "SupportVectorMachine"]

model_dict = {
        "DecisionTree": tree.DecisionTreeClassifier(),
        "LogisticRegression": LogisticRegression(),
        "RandomForest": RandomForestClassifier(max_depth=3),
        "SupportVectorMachine": svm.SVC(),
        }

def get_data(path: str) -> list[tuple]:
    ret_arr = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        print(file_path, end="\r")

        array = np.load(file_path)
        array = array.flatten()
        array = array[:SIZE]
        array = list(array)
        ret_arr.append((array, 0 if path.endswith('benigns') else 1))
        
    return ret_arr

def main():
    data = get_data(BENIGN_PATH)
    data2 = get_data(MALWARE_PATH)
    data.extend(data2)
    random.shuffle(data)

    X = [d[0] for d in data]
    y = [d[1] for d in data]

    for model_name in models:
        model = model_dict[model_name]
        
        model = model.fit(X, y)
        
        with open(os.path.join(os.curdir, model_name + '.pickle'), "wb") as f:
            pickle.dump(model, f)

if __name__ == "__main__":
    main()

