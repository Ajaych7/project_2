import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class VoiceCloneModel(nn.Module):
    def __init__(self, input_dim=80, embedding_dim=256):
        super(VoiceCloneModel, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, embedding_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=4)
        
        # Decoder
        self.decoder = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        
        # Final projection
        self.proj = nn.Linear(embedding_dim, input_dim)
    
    def forward(self, x):
        # x shape: (batch, time, features)
        x = x.permute(0, 2, 1)  # Conv1d expects (batch, channels, time)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)  # Back to (batch, time, channels)
        
        # Self-attention
        x = x.permute(1, 0, 2)  # (time, batch, channels) for attention
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 0, 2)  # Back to (batch, time, channels)
        
        # Decoder
        x, _ = self.decoder(x)
        
        # Project back to feature space
        x = self.proj(x)
        return x

# Initialize model
model = VoiceCloneModel()
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
