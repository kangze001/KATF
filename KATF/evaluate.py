import torch
import numpy as np
from torch.utils.data import DataLoader
from data.dataset_processor import NetworkFlowDataset
from models.KATF import KATF 
from config import Config
from data.split_data import get_dataset_paths
from sklearn.metrics import classification_report
import os
from tqdm import tqdm

def evaluate_model(model_path, dataset_type="test"):
    """
    Loads a saved PyTorch model and evaluates its performance 
    on the specified dataset (test or validation).

    Parameters:
    model_path (str): Path to the model checkpoint file.
    dataset_type (str): Type of dataset to evaluate ('test' or 'valid').
    """
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[⚙️] Using device: {device}")
    
    # Get dataset paths
    dataset_paths = get_dataset_paths()
    
    # Determine the dataset path to evaluate
    if dataset_type == "test":
        dataset_path = dataset_paths["test"]
    elif dataset_type == "valid":
        dataset_path = dataset_paths["valid"]
    else:
        print(f"[⚠️] Warning: Unknown dataset type '{dataset_type}'. Using test set.")
        dataset_path = dataset_paths["test"]
    
    # Load dataset
    # Note: dataset_path is redundantly set to dataset_paths["test"] in the original, fixed here.
    dataset = NetworkFlowDataset(
        npz_path=dataset_path,
        use_length=config.USE_LENGTH,
        num_channels=6,
        verbose=config.VERBOSE
    )
    
    # Create Data Loader
    loader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Initialize Model
    # Note: verbose=False for evaluation initialization to prevent print clutter
    model = KATF(
        input_channels=6,
        sequence_length=config.USE_LENGTH,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        num_classes=dataset.num_classes,
        device=device,
        verbose=False 
    )
    
    # Load model weights
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'N/A')
        print(f"[📦] Model loaded successfully: {model_path} (Epoch {epoch})")
    else:
        print(f"[❌] Error: Model file not found - {model_path}")
        return
    
    model.to(device)
    model.eval()
    
    # --- Evaluation Loop ---
    all_labels = []
    all_preds = []
    
    test_iter = tqdm(loader, desc=f"Evaluating {dataset_type} dataset", unit="batch", dynamic_ncols=True)
    
    with torch.no_grad():
        for inputs, labels in test_iter:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    # Calculate Metrics
    all_labels_arr = np.array(all_labels)
    all_preds_arr = np.array(all_preds)
    
    acc = (all_preds_arr == all_labels_arr).sum() / len(all_labels_arr)
    report = classification_report(
        all_labels_arr, 
        all_preds_arr, 
        target_names=[f"class_{i}" for i in range(dataset.num_classes)],
        output_dict=False,
        digits=4, # Display 4 decimal places
        zero_division=0
    )
    
    # Print Results
    print("\n" + "=" * 60)
    print(f"[🏆] Performance Evaluation on {dataset_type} Set (Samples: {len(dataset)})")
    print(f"    Accuracy: {acc*100:.4f}%")
    print(f"    Classification Report:\n{report}")
    print("=" * 60)
    
    # Save Results
    result_file = f"{os.path.splitext(model_path)[0]}_{dataset_type}_results.txt"
    with open(result_file, "w") as f:
        f.write(f"{dataset_type} Set Accuracy: {acc*100:.4f}%\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"[💾] Evaluation results saved to: {result_file}")

if __name__ == "__main__":
    # Example usage: Replace 'best_katf.pth' with your actual model path
    model_path = "best_KATF.pth" 
    evaluate_model(model_path, dataset_type="test")