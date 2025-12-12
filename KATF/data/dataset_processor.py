import numpy as np
import torch
from torch.utils.data import Dataset

class NetworkFlowDataset(Dataset):
    """
    A PyTorch Dataset for loading and preprocessing network flow data from an NPZ file.
    The data is reshaped and normalized to fit a [batch, channels, sequence] format.
    """
    def __init__(self, npz_path, use_length=1950, num_channels=6):
        self.num_channels = num_channels
        
        print(f"[📁] Loading dataset: {npz_path}")
        
        data = np.load(npz_path)
        
        # Load and slice data
        # [B, C_orig, T_orig] -> [B, C_orig, use_length] (C_orig is often 1 here)
        self.X = data['X'][:, :, :use_length]
        self.y = data['y']
        
        # Fix and adjust data dimensions (simplified)
        if len(self.X.shape) == 2:
            # Add channel dimension: [B, T] -> [B, 1, T]
            self.X = self.X[:, np.newaxis, :]
        elif len(self.X.shape) > 3:
            # Squeeze extra dimensions if present (e.g., [B, 1, 1, T] -> [B, 1, T])
             self.X = np.squeeze(self.X)
        
        # Ensure we have the channel dimension (e.g., [B, 1, T])
        if self.X.shape[1] == 1:
            # Expand single-channel data to required num_channels: [B, 1, T] -> [B, num_channels, T]
            self.X = np.repeat(self.X, num_channels, axis=1)

        self.num_classes = len(np.unique(self.y))
        
        print(f"[🔍] Dataset stats: {len(self.X)} samples, {self.num_classes} classes")
        print(f"[📐] Data shape: {self.X.shape} (batch x channels x sequence)")
        
        self._normalize()
        
        print(f"[✅] Dataset loaded and preprocessed!")
    
    def _normalize(self):
        """
        Applies magnitude normalization ([-1, 1]) to each channel of each sample.
        Normalization formula: sign(x) * (|x| / max(|x|))
        """
        print(f"[🔧] Starting normalization...")
        
        # Normalize each sample/channel pair
        for i in range(len(self.X)):
            for ch in range(self.num_channels):
                channel_data = self.X[i, ch, :]
                
                # Normalization parameters
                sign = np.sign(channel_data)
                abs_val = np.abs(channel_data)
                
                # Avoid division by zero
                max_val = np.max(abs_val)
                if max_val == 0:
                    max_val = 1.0 
                
                # Apply normalization: scaled to [-1, 1]
                scaled = abs_val / max_val
                self.X[i, ch, :] = sign * scaled
        
        print(f"[✅] Normalization complete")
    
    def __len__(self):
        """Returns the total number of samples."""
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Returns the sample data and label at the given index.
        Data shape is [channels, sequence], as required for PyTorch models.
        """
        # Convert numpy array to torch tensor
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)