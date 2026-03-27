import torch
import torch.nn as nn

class RobustNeuralKalmanFilter(nn.Module):
    """
    Neural Kalman Filter module (NKF)
    Implements the full recursive update of P, K, Q, R
    to refine the hidden states of the Bi-LSTM.
    """
    def __init__(self, hidden_dim):
        super(RobustNeuralKalmanFilter, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Noise prediction network: dynamically estimates process noise Q and measurement noise R
        self.noise_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Softplus()  # ensures positive variance and numerical stability
        )

    def forward(self, lstm_out):
        batch_size, seq_len, h_dim = lstm_out.size()
        
        # 1. Dynamically predict noise parameters for all time steps
        noise_params = self.noise_net(lstm_out)
        Q_all = noise_params[:, :, :h_dim]
        R_all = noise_params[:, :, h_dim:]
        
        # 2. Initialize state estimation and covariance
        # Initial state is set to the first LSTM output, initial uncertainty P = 0.1
        state = lstm_out[:, 0, :].clone()
        P = torch.ones_like(state) * 0.1 
        
        filtered_states = []
        
        for t in range(seq_len):
            # Use current LSTM output as observation z
            z = lstm_out[:, t, :]
            
            # --- Prediction step ---
            # x_pred = x (F = I)
            # P_pred = P + Q
            P_pred = P + Q_all[:, t, :]
            
            # --- Update step ---
            # Kalman gain K = P_pred / (P_pred + R)
            K = P_pred / (P_pred + R_all[:, t, :] + 1e-6)
            
            # State update: x = x_pred + K * (z - x_pred)
            state = state + K * (z - state)
            
            # Covariance update: P = (1 - K) * P_pred
            P = (1.0 - K) * P_pred
            
            filtered_states.append(state)
            
        return torch.stack(filtered_states, dim=1)


class KATFModel(nn.Module):
    """
    Full KATF model with ablation switches:
    - use_kalman: enable/disable neural Kalman state refinement
    - use_attention: enable/disable attention mechanism
                     (if disabled, global average pooling is used)
    """
    def __init__(self, input_channels=6, sequence_length=1950, hidden_dim=256, 
                 num_layers=3, num_classes=4, device='cuda',
                 use_kalman=True, use_attention=True):
        super(KATFModel, self).__init__()
        self.device = device
        self.use_kalman = use_kalman
        self.use_attention = use_attention
        print(f"[📦] Model loaded: {use_kalman} {use_attention}")
        
        # 1. Feature extraction layer (CNN stack)
        # Note: sequence_length default is set to 3000 according to your config
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_channels, 48, kernel_size=25, stride=5, padding=12),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Conv1d(48, 96, kernel_size=15, stride=3, padding=7),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Conv1d(96, 192, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Conv1d(192, 384, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(384),
            nn.ReLU()
        )
        
        # 2. Temporal modeling layer (Bi-LSTM)
        self.lstm = nn.LSTM(
            input_size=384,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            # dropout=0.2 if num_layers > 1 else 0  # optional inter-layer dropout
        )
        
        # 3. Ablation component: Neural Kalman Filter
        if self.use_kalman:
            self.kalman_filter = RobustNeuralKalmanFilter(hidden_dim * 2)
        
        # 4. Ablation component: Attention mechanism
        if self.use_attention:
            self.attention_net = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
        
        # 5. Classification layer
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),  # widened classifier
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.to(device)

    def forward(self, x):
        # Input shape: (batch, input_channels, seq_len)
        
        # CNN feature extraction
        feat = self.feature_extractor(x)
        feat = feat.permute(0, 2, 1)  # reshape to (batch, seq, features)
        
        # Bi-LSTM modeling
        x, _ = self.lstm(feat) 
        
        # --- Ablation logic 1: Kalman refinement ---
        if self.use_kalman:
            x = self.kalman_filter(x)
        
        # --- Ablation logic 2: feature aggregation ---
        if self.use_attention:
            # Attention-weighted aggregation
            attn_weights = torch.softmax(self.attention_net(x), dim=1)
            context = torch.sum(attn_weights * x, dim=1)
        else:
            # Baseline: global average pooling
            context = torch.mean(x, dim=1)
            
        # Classification output
        return self.classifier(context)


# ==========================================
# Example usage in training script:
# ==========================================
"""
# 1. Full KATF model
model = KATFModel(use_kalman=True, use_attention=True)

# 2. Ablation: without Kalman module (w/o Kalman)
model = KATFModel(use_kalman=False, use_attention=True)

# 3. Ablation: without Attention (w/o Attention)
model = KATFModel(use_kalman=True, use_attention=False)

# 4. Baseline model (Bi-LSTM only)
model = KATFModel(use_kalman=False, use_attention=False)
"""
