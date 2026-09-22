import numpy as np
from tqdm import tqdm
import os

# ============ Settings ============
GLOVE_PATH = "./data/glove.6B/glove.6b.50d.txt"
SEED = 42
SPLIT_RATIO = 0.75
OUTPUT_DIR = "./data/glove_6b_50d_split"



# =========== Functions ============
def set_seed(seed):
    np.random.seed(seed)

def save_vectors(filename, vectors):
    with open(filename, 'w', encoding='utf-8') as f:
        for vector in vectors:
            f.write(vector)
    print(f"Saved: {filename}")

def load_glove_embeddings(glove_path, dim=50):
    lines = []
    with open(glove_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    np.random.shuffle(lines)  # shuffle for randomness
    return lines


def main():
    set_seed(SEED)

    # Load GloVe embeddings
    lines = load_glove_embeddings(GLOVE_PATH)

    total = len(lines)
    split = int(SPLIT_RATIO * total)
    data_lines, query_lines = lines[:split], lines[split:]

    print("Data lines count:", len(data_lines))
    print("Query lines count:", len(query_lines))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Save split text files
    save_vectors(os.path.join(OUTPUT_DIR, "X_data.txt"), data_lines)
    save_vectors(os.path.join(OUTPUT_DIR, "X_query.txt"), query_lines)


if __name__ == "__main__":
    main()
