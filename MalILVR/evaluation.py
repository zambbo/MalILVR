import os
import numpy as np
import argparse
from tqdm import tqdm
import pickle
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

SAMPLE_BASE_PATH="./generated_samples/"
SAMPLE_PATH = os.path.join(SAMPLE_BASE_PATH, "plain_benign_npy")
BLACKBOX_PATH = "../blackbox/LogisticRegression.pickle"
LOG_PATH = "./log_mal_ilvr/LogitsticRegression"

def argument_parse():
    parser = argparse.ArgumentParser(description="Evaluation Argument Parser")

    parser.add_argument('--sample_path', action='store', default=SAMPLE_PATH, required=True, metavar=SAMPLE_PATH, help="directory that stores npy binary vectors")
    parser.add_argument('--blackbox_path', action='store', default=BLACKBOX_PATH, required=True, metavar=BLACKBOX_PATH, help="directory taht stores blackbox model")
    parser.add_argument('--log_path', action='store', default=LOG_PATH, metavar=LOG_PATH, help="directory that log information about evaluation")

    parser.add_argument('--name', action='store', default="model_name", metavar="model_name", help="Model name")

    args = parser.parse_args()

    return args



class Evaluator:
    def __init__(self, args):
        self.sample_path = args.sample_path
        self.blackbox_path = args.blackbox_path
        self.model_name = args.name
        print(f"[+] sample... {self.sample_path}")
        print(f"[+] blackbox... {self.blackbox_path}")
        print(f"[+] model name... {self.model_name}")
        self.log_path = args.log_path

        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        self.log_f = open(os.path.join(self.log_path, self.model_name + '_log.txt'), "wt")
        
        self.load_blackbox()
        self.load_vectors()

    '''
    Evaluate Diversity
    '''

    def load_vectors(self):
        self.vectors = []
        print("[+] Loading Vectors...")
        for file in tqdm(os.listdir(self.sample_path)):
            file_path = os.path.join(self.sample_path, file)
            
            array = np.load(file_path)
            array = list(array)
            self.vectors.append(array)

    def load_blackbox(self):
        f = open(self.blackbox_path, "rb")
        self.blackbox = pickle.load(f)
        f.close()

    def average_hamming_distance(self, batch_size=1000) -> int:
        vectors = self.vectors
        num_vectors = len(vectors)
        vector_length = len(vectors[0])
        print("[+] Calculating hamming distance...")
        self.log_f.write("[+] Calculating hamming distance...\n")

        total_distance = 0
        num_pairs = 0

        for i in tqdm(range(0, num_vectors, batch_size)):
            start = i
            end = min(i + batch_size, num_vectors)

            # Convert the batch of vectors into a NumPy array
            batch_array = np.array(vectors[start:end])

            # Reshape the array to have dimensions (batch_size, 1, vector_length)
            reshaped_array = batch_array.reshape(end - start, 1, vector_length)

            for j in tqdm(range(num_vectors)):
                # Reshape the individual vector to match the shape of the tiled array
                vector = np.array(vectors[j]).reshape(1, 1, vector_length)

                # Tile the individual vector to create a matrix of shape (batch_size, 1, vector_length)
                tiled_vector = np.tile(vector, (end - start, 1, 1))

                # Compute the element-wise XOR between the batch and the individual vector
                xor_matrix = np.bitwise_xor(reshaped_array, tiled_vector)

                # Count the number of set bits (ones) in the XOR matrix along the vector_length axis
                set_bits_count = np.sum(xor_matrix, axis=2)

                # Accumulate the distance and number of pairs
                total_distance += np.sum(set_bits_count)
                num_pairs += (end - start)

        # Calculate the average Hamming distance
        average_distance = total_distance / (num_pairs * vector_length)
        self.log_f.write(str(average_distance) + '\n')
        return average_distance

    def cal_tpr(self):
        predicted = self.blackbox.predict(self.vectors)
        
        print("[+] Calc True Positive Rate...")
        self.log_f.write("[+] Calc True Positive Rate...\n")

        predict_malware = len([vec for vec in predicted if vec == 1])
        all_vectors = len(self.vectors)
        
        tpr = predict_malware / all_vectors
        self.log_f.write(str(tpr) + '\n')
        return tpr

    def visualize_clustering(self):
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(self.vectors)

        plt.figure()
        plt.scatter(data_pca[:, 0], data_pca[:, 1])
        plt.title(self.model_name + ' PCA')

        plt.savefig(os.path.join(self.log_path, self.model_name + ".png"))


if __name__ == "__main__":
    args = argument_parse()
    evaluator = Evaluator(args)
    print(evaluator.cal_tpr())
    print(evaluator.average_hamming_distance(batch_size=1000))
    evaluator.visualize_clustering()
