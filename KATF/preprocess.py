from data.split_data import split_dataset
from config import Config

def preprocess_data():
    """
    Main function for data preprocessing, which currently involves 
    splitting the raw dataset into training, validation, and test sets.
    """
    config = Config()
    
    # Execute the data splitting function
    split_dataset()
    
    # Report completion status and file paths
    print("\n[✅] Data Preprocessing Complete! Datasets saved to:")
    print(f"    - Train Set: {config.TRAIN_DATA_PATH}")
    print(f"    - Validation Set: {config.VALID_DATA_PATH}")
    print(f"    - Test Set: {config.TEST_DATA_PATH}")

if __name__ == "__main__":
    preprocess_data()