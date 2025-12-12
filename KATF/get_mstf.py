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
    
    # ==================================
    # 1. Configuration and Reproducibility
    # ==================================
    
    # Set a fixed seed for reproducibility
    FIXED_SEED = 2024
    random.seed(FIXED_SEED)
    torch.manual_seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)

    # --- File and Data Configuration ---
    DATASET_DIR = "dataset"
    
    # IMPORTANT: Update the raw and output file names below
    RAW_DATA_FILENAME = "specific_dataset_name.npz"
    OUTPUT_DATA_FILENAME = "specific_dataset_name_mstf.npz"
    
    SEQUENCE_LENGTH = 10000  # The target length for alignment
    
    # --- Path Construction ---
    IN_PATH = os.path.join(DATASET_DIR)
    RAW_FILE = os.path.join(DATASET_DIR, RAW_DATA_FILENAME)
    OUT_FILE = os.path.join(DATASET_DIR, OUTPUT_DATA_FILENAME)

    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(f"The dataset directory does not exist: {IN_PATH}")

    # ==================================
    # 2. Processing Logic
    # ==================================
    
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
        # All samples in X are padded or truncated to SEQUENCE_LENGTH
        X = data_processor.length_align(X, SEQUENCE_LENGTH)
        
        # 2. Extract the Multi-Scale Temporal Fingerprinting (MSTF)
        # ---------------------------------------------------------------------
        '''
        Note on MSTF Feature Usage:
        
        For offline testing or real-time deployment using our self-built 
        Bot Traffic Dataset, please use the MSTF (Multi-Scale Traffic Temporal 
        Fingerprinting) features for training and inference. This is necessary 
        to validate the real-time testing capabilities of our KATF model.
        
        However, for public datasets such as CW, OW, WTF-PAD, Front, 
        Walkie-Talkie, and TrafficSliver, you must not use the MSTF features, 
        as they will not be suitable. For these public datasets, only the raw 
        time-series features should be used.
        '''
        # ---------------------------------------------------------------------
        X = data_processor.extract_MSTF(X)
        
        # Print processing information
        print(f"[✅] MSTF Feature Extraction Done.")
        print(f"    - Initial Shape: {initial_shape}")
        print(f"    - Final Shape: X={X.shape}, y={y.shape}")
        
        # Save the processed data
        np.savez(OUT_FILE, X=X, y=y)
        print(f"[💾] Processed data saved to: {OUT_FILE}")
    else:
        # Print a message if the output file already exists
        print(f"[ℹ️] Output file already exists, skipping generation: {OUT_FILE}")

if __name__ == '__main__':
    main()
