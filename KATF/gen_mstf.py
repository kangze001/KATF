import numpy as np
import os
import random
import torch
import data_processor 

def main():
    """
    Generates the Multi-Scale Traffic Aggregation Matrix (MSTF) features 
    from raw time series data and saves the result to an .npz file.
    """
    # Set a fixed seed for reproducibility
    FIXED_SEED = 2024
    random.seed(FIXED_SEED)
    torch.manual_seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)

    # --- Configuration ---
    DATASET_DIR = "dataset"
    # IMPORTANT: Update the raw and output file names
    RAW_DATA_FILENAME = "specific_dataset_name.npz"
    OUTPUT_DATA_FILENAME = "specific_dataset_name_mstf.npz"
    SEQUENCE_LENGTH = 10000 # The target length for alignment
    
    IN_PATH = os.path.join(DATASET_DIR)
    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(f"The dataset directory does not exist: {IN_PATH}")

    RAW_FILE = os.path.join(DATASET_DIR, RAW_DATA_FILENAME)
    OUT_FILE = os.path.join(DATASET_DIR, OUTPUT_DATA_FILENAME)

    # If the output file does not exist, process the input file
    if not os.path.exists(OUT_FILE):
        print(f"[⏳] Starting feature extraction from: {RAW_FILE}")
        
        # Load dataset
        try:
            data = np.load(RAW_FILE)
            X = data["X"]
            y = data["y"]  
        except FileNotFoundError:
             raise FileNotFoundError(f"Raw data file not found: {RAW_FILE}")

        initial_shape = X.shape

        # 1. Align the sequence length
        X = data_processor.length_align(X, SEQUENCE_LENGTH)
        
        # 2. Extract the Multi-Scale Traffic Aggregation Matrix (MSTF/TAM)
        X = data_processor.extract_MSTF(X)
        
        # Print processing information
        print(f"[✅] MSTF Feature Extraction Done.")
        print(f"    - Initial Shape: {initial_shape}")
        print(f"    - Final Shape: X={X.shape}, y={y.shape}")
        
        # Save the processed data
        np.savez(OUT_FILE, X=X, y=y)
        print(f"[💾] Processed data saved to: {OUT_FILE}")
    else:
        # Print a message if the output file already exists
        print(f"[ℹ️] Output file already exists, skipping generation: {OUT_FILE}")

if __name__ == '__main__':
    main()