import os
import numpy as np
import argparse


SAMPLE_BASE_PATH="./generated_samples/"
SAMPLE_PATH = os.path.join(SAMPLE_BASE_PATH, "plain_benign_npy")

def argument_parse():
    parser = argparse.ArgumentParser(description="Evaluation Argument Parser")

    parser.add_argument('--sample_path', action='store', default=SAMPLE_PATH, required=True, metavar=SAMPLE_PATH, help="directory that stores npy binary vectors")
    
    args = parser.parse_args()

    return args



class Evaluator:
    def __init__(self, sample_path):
        self.sample_path = sample_path
    '''
    Evaluate Diversity
    '''
    def hamming_distance(vec1: np.array, vec2: np.array) -> int:
        if (len(vec1) != len(vec2)):
            print("Length shoulde be same")
            return

        return sum(c1 != c2 for c1, c2 in zip(vec1, vec2))

    def average_hamming_distance(vectors: list[np.array]) -> int:
        total_distances = 0
        num_pairs = 0

        for i in range(len(vectors)):
            for j in range(i+1, len(vectors)):
                distance = hamming_distance(vectors[i], vectors[j])
                total_distance += distance
                num_pairs += 1

        average_distance = total_distance / num_pairs
        return average_distance

if __name__ == "__main__":
    args = argument_parse()
