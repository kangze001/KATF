import torch
import torch.nn as nn
import math

class KATF(nn.Module):
    """
    KATF Model for flow classification.
    The model combines 1D CNN for feature extraction, Bi-LSTM for sequence modeling,
    a Kalman Net for state estimation/filtering, and an Attention mechanism for final context vector generation.
    """
    def __init__(self, input_channels=6, sequence_length=1950, hidden_dim=256, 
                 num_layers=3, num_classes=4, device='cuda', verbose=True):
        super(KATF, self).__init__()
        self.device = device
        self.verbose = verbose
        self.conv_length = None
        
        # 1. Feature Extraction (1D CNN)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_channels, 48, kernel_size=25, stride=5, padding=12),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Conv1d(48, 96, kernel_size=15, stride=3, padding=7),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Conv1d(96, 192, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Conv1d(192, 384, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(384),
            nn.ReLU()
        )

        # Calculate feature sequence length after CNN layers
        def calc_output_len(L_in, k, s, p):
            return math.floor((L_in + 2 * p - k) / s) + 1

        self.conv_length = sequence_length
        self.conv_length = calc_output_len(self.conv_length, 25, 5, 12)
        self.conv_length = calc_output_len(self.conv_length, 15, 3, 7)
        self.conv_length = calc_output_len(self.conv_length, 7, 2, 3)
        self.conv_length = calc_output_len(self.conv_length, 5, 1, 2)
        
        if self.verbose:
            print(f"[📏] Sequence Compression: {sequence_length} → {self.conv_length} (Ratio: {sequence_length/self.conv_length:.1f}x)")
        
        # 2. Sequence Modeling (Bidirectional LSTM)
        self.lstm = nn.LSTM(
            input_size=384,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # 3. Kalman Network (outputs three parameters: alpha, log(Q), log(R))
        self.kalman_net = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim*4),
            nn.ReLU(),
            nn.Linear(hidden_dim*4, 3) # alpha (gain), log(Q), log(R)
        )
        
        # 4. Attention Mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 5. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self._init_weights()
        self.to(device)
        
        if verbose:
            print(f"[🧠] KATF Model Initialization Complete - Device: {device}")
            print(f"    - Hidden Dimension: {hidden_dim}")
            print(f"    - LSTM Layers: {num_layers}")
            print(f"    - Number of Classes: {num_classes}")
    
    def _init_weights(self):
        """Initializes model weights using Xavier (Linear) and Kaiming (Conv1d) initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 1. Feature Extraction: (B, C, L) -> (B, 384, L')
        features = self.feature_extractor(x)
        # Permute to (B, L', C') for LSTM input
        features = features.permute(0, 2, 1) 
        
        # 2. LSTM: (B, L', C') -> (B, L', 2*H)
        lstm_out, _ = self.lstm(features)
        
        # 3. Kalman Net: (B, L', 2*H) -> (B, L', 3)
        kalman_params = self.kalman_net(lstm_out)
        
        # Extract Kalman parameters
        alpha = torch.sigmoid(kalman_params[..., 0]) # Kalman Gain scaling
        Q = torch.exp(kalman_params[..., 1]) # Process Noise Covariance (variance)
        R = torch.exp(kalman_params[..., 2]) # Measurement Noise Covariance (variance)
        
        # 4. Kalman Filter Loop (Simplified State Space Model)
        
        # Initial state is the first LSTM output, ensures the state vector size is correct (2*H)
        # Note: This is an approximation of the Measurement Update and Time Update of a full Kalman Filter
        state = lstm_out[:, 0, :].clone() 
        states = []
        
        for t in range(self.conv_length):
            # Prediction Step (Approx): The next state is predicted to be the current state
            predicted_state = state 
            # Measurement (LSTM output)
            measurement = lstm_out[:, t, :] 
            
            # Residual (Innovation)
            residual = measurement - predicted_state
            
            # Denominator (R + Q)
            denominator = Q[:, t].unsqueeze(1) + R[:, t].unsqueeze(1)
            denominator = torch.clamp(denominator, min=1e-6) # Stability check
            
            # Kalman Gain (K = alpha * Q / (Q + R))
            # We use alpha (learned scale factor) on the calculated gain
            K = alpha[:, t].unsqueeze(1) * (Q[:, t].unsqueeze(1) / denominator) 
            
            # Update Step: state = predicted_state + K * residual
            state = predicted_state + K * residual 
            states.append(state)
        
        # Kalman Filtered Output: (B, L', 2*H)
        kalman_out = torch.stack(states, dim=1)
        
        # 5. Attention Mechanism
        # Compute weights: (B, L', 2*H) -> (B, L', 1)
        attention_weights = torch.softmax(self.attention(kalman_out), dim=1) 
        # Weighted sum: (B, L', 2*H) * (B, L', 1) -> (B, 2*H)
        context = torch.sum(attention_weights * kalman_out, dim=1) 
        
        # 6. Classifier: (B, 2*H) -> (B, num_classes)
        return self.classifier(context)