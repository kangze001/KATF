import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from data.dataset_processor import NetworkFlowDataset
from models.KATF import KATF
from training.trainer import ModelTrainer
from config import Config
from data.split_data import get_dataset_paths
from torch.optim.lr_scheduler import ReduceLROnPlateau

def main():
    """
    Main function to setup and run the model training process.
    """
    config = Config()
    
    # Set fixed seeds for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[⚙️] Using device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[⚡] GPU Model: {gpu_name}")
        print(f"[💾] Total GPU Memory: {total_mem:.2f}GB")
    
    # Get dataset paths
    dataset_paths = get_dataset_paths()
    
    # --- Data Initialization ---
    print(f"[🔍] Initializing datasets...")
    train_dataset = NetworkFlowDataset(
        npz_path=dataset_paths["train"],
        use_length=config.USE_LENGTH,
        num_channels=6,
        verbose=config.VERBOSE
    )
    valid_dataset = NetworkFlowDataset(
        npz_path=dataset_paths["valid"],
        use_length=config.USE_LENGTH,
        num_channels=6,
        verbose=config.VERBOSE
    )
    print(f"[✅] Dataset Initialization Complete. Classes: {train_dataset.num_classes}")
    
    
    # --- Data Loaders ---
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    
    # --- Model Initialization ---
    model = KATF(
        input_channels=6,
        sequence_length=config.USE_LENGTH,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        num_classes=train_dataset.num_classes,
        device=device,
        verbose=config.VERBOSE
    )
    
    # Calculate and print model parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[🧠] Total trainable parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Loss function and Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE, # Use config LR
        weight_decay=1e-5
    )
    # LR Scheduler (reduces LR when validation metric stops improving)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=3,
        min_lr=1e-5,
        verbose=True
    )
    
    # --- Trainer Initialization ---
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        model_name=config.MODEL_NAME,
        num_classes=train_dataset.num_classes,
        verbose=config.VERBOSE
    )
    
    # Set gradient accumulation steps
    trainer.set_gradient_accumulation(config.GRADIENT_ACCUM_STEPS)
    
    # --- Start Training ---
    best_acc, best_f1 = trainer.train(config.NUM_EPOCHS)
    
    print(f"\n[🏁] Training complete! Best Validation Accuracy: {best_acc*100:.2f}%")
    print(f"[💾] Best model saved as: best_{config.MODEL_NAME}.pth")
    
    # Performance analysis
    final_mem_usage = max(trainer.gpu_mem_usage) if trainer.gpu_mem_usage else 0
    print("\n[📈] GPU Memory Usage Analysis:")
    print(f"    - Peak Memory Used: {final_mem_usage:.2f}GB")
    print(f"    - Configured Max Memory: {config.MAX_MEMORY_USAGE}GB")
    if config.MAX_MEMORY_USAGE > 0:
        print(f"    - Utilization Ratio: {final_mem_usage/config.MAX_MEMORY_USAGE*100:.1f}%")

if __name__ == "__main__":
    main()