import os

class Config:
    """
    Configuration class for the flow classification project, defining
    hyperparameters, data paths, and training settings.
    """
    # --- Hyperparameters & Model Settings ---
    SEED = 42
    BATCH_SIZE = 256
    LEARNING_RATE = 2e-4
    HIDDEN_DIM = 256     # Hidden dimension for LSTM
    NUM_LAYERS = 3       # Number of LSTM layers
    NUM_EPOCHS = 30
    USE_LENGTH = 1950    # Sequence length for the time series input
    MODEL_NAME = "KATF"  # Model identifier
    
    # --- Data Paths Configuration ---
    DATA_DIR = "./dataset"
    # IMPORTANT: Update the dataset name below
    RAW_DATA_FILENAME = "specific_dataset_name.npz" 
    RAW_DATA_PATH = os.path.join(DATA_DIR, RAW_DATA_FILENAME)
    TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.npz")
    VALID_DATA_PATH = os.path.join(DATA_DIR, "valid.npz")
    TEST_DATA_PATH = os.path.join(DATA_DIR, "test.npz")
    
    # --- Training & System Settings ---
    VERBOSE = True
    NUM_WORKERS = 0             # Dataloader workers (set to 0 for Windows/debugging)
    GRADIENT_ACCUM_STEPS = 1    # Gradient accumulation steps (1 for off)
    MAX_MEMORY_USAGE = 22.0     # Max VRAM (in GB) to limit batch size dynamically (optional feature)
    
    # --- Data Split Ratios ---
    TRAIN_RATIO = 0.8  # 80% for training
    VALID_RATIO = 0.1  # 10% for validation
    TEST_RATIO = 0.1   # 10% for testing

    # --- Derived Attributes (Optional, but useful for verification) ---
    @property
    def TOTAL_RATIO(self):
        return self.TRAIN_RATIO + self.VALID_RATIO + self.TEST_RATIO
    
    def __init__(self):
        # Basic validation check
        if not (self.TRAIN_RATIO + self.VALID_RATIO + self.TEST_RATIO) == 1.0:
            raise ValueError("Data split ratios must sum to 1.0 (100%).")