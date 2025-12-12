import numpy as np
import os
from sklearn.model_selection import train_test_split
from config import Config

def split_dataset():
    """
    Loads raw data, splits it into training, validation, and test sets
    based on ratios defined in Config, and saves the split datasets.
    """
    config = Config()
    
    # ℹ️ Start dataset splitting
    print(f"[INFO] Starting dataset split: {config.RAW_DATA_PATH}")
    print(f"Split Ratios: Train {config.TRAIN_RATIO*100}%, Valid {config.VALID_RATIO*100}%, Test {config.TEST_RATIO*100}%")
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(config.TRAIN_DATA_PATH), exist_ok=True)
    
    # Load raw data
    data = np.load(config.RAW_DATA_PATH)
    X = data['X']
    y = data['y']
    
    # Step 1: Split into Training and Temporary set
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=config.VALID_RATIO + config.TEST_RATIO,
        random_state=config.SEED,
        stratify=y
    )
    
    # Step 2: Split Temporary set into Validation and Test sets
    # The new test_size is the proportion of the test set *within* the temporary set
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp,
        test_size=config.TEST_RATIO / (config.VALID_RATIO + config.TEST_RATIO),
        random_state=config.SEED,
        stratify=y_temp
    )
    
    # Save the split datasets
    np.savez(config.TRAIN_DATA_PATH, X=X_train, y=y_train)
    np.savez(config.VALID_DATA_PATH, X=X_valid, y=y_valid)
    np.savez(config.TEST_DATA_PATH, X=X_test, y=y_test)
    
    # ✅ Dataset splitting complete
    print(f"[SUCCESS] Dataset splitting complete!")
    print(f"    - Train Set: {len(X_train)} samples - {config.TRAIN_DATA_PATH}")
    print(f"    - Valid Set: {len(X_valid)} samples - {config.VALID_DATA_PATH}")
    print(f"    - Test Set: {len(X_test)} samples - {config.TEST_DATA_PATH}")

def get_dataset_paths():
    """
    Returns a dictionary of paths to the split datasets.
    """
    config = Config()
    return {
        "train": config.TRAIN_DATA_PATH,
        "valid": config.VALID_DATA_PATH,
        "test": config.TEST_DATA_PATH
    }