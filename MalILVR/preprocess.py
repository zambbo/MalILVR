import numpy as np
import os

arr = ["api_vectors", "benign", "malware"]

def do(name):
    a = os.listdir(f"./datasets/{name}")

    for aa in a:
        item = np.load(f"./datasets/{name}/" + aa).reshape(64,64,1)
        np.save(f"./datasets/{name}/" + aa, item)

for name in arr:
    do(name)
