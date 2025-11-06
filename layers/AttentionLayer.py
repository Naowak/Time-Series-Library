import torch
import torch.nn as nn
import math


class AttentionLayer(nn.Module):

    def __init__(self, d_model, num_heads=8, dropout=0.1, self_attention=True, dtype=torch.float32, device='cpu'):
        """Initialize the AttentionLayer module.
            
        Args:
            d_model: Model dimension (D)
            num_heads: Number of attention heads
            dropout: Dropout rate
            self_attention: If True, use self-attention; else, use encoder-decoder attention
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model doit être divisible par num_heads"
        # Store hyper-parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension par tête
        self.self_attention = self_attention
        
        # Linear projection for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, device=device, dtype=dtype) # [D, D]
        self.W_k = nn.Linear(d_model, d_model, device=device, dtype=dtype) # [D, D]
        self.W_v = nn.Linear(d_model, d_model, device=device, dtype=dtype) # [D, D]

        # Output projection
        self.W_o = nn.Linear(d_model, d_model, device=device, dtype=dtype) # [D, D]

        # Dropout & scaling factor
        self.dropout_layer = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x, state=None):
        """
        Args:
            x: Input tensor of shape (B, T, M, D)
            state: Optional state tensor for encoder-decoder attention of shape (B, T, M, D)
        Returns:
            output: Tensor of shape (B, T, M, D)
        """
        # Check state (not used in self-attention)
        if state is None and not self.self_attention:
            raise ValueError("State must be provided for non-self-attention.")

        batch_size, seq_len, units, _ = x.shape

        # Linear projections
        if self.self_attention:
            Q = self.W_q(x) # (B, T, M, D)
            K = self.W_k(x) # (B, T, M, D)
            V = self.W_v(x) # (B, T, M, D)
        else:
            Q = self.W_q(x) # (B, T, M, D)
            K = self.W_k(state) # (B, T, M, D)
            V = self.W_v(state) # (B, T, M, D)

        # Reshape for multi-head: (B, T, H, M, D)
        Q = Q.view(batch_size, seq_len, units, self.num_heads, self.d_k).transpose(2, 3) # (B, T, H, M, d_k)
        K = K.view(batch_size, seq_len, units, self.num_heads, self.d_k).transpose(2, 3) # (B, T, H, M, d_k)
        V = V.view(batch_size, seq_len, units, self.num_heads, self.d_k).transpose(2, 3) # (B, T, H, M, d_k)

        # Compute attention scores: Q @ K^T / sqrt(d_k)
        scores = Q @ K.transpose(-2, -1) / self.scale # (B, T, H, M, M)

        # Softmax along the last dimension (M)
        attn_weights = torch.softmax(scores, dim=-1) # (B, T, H, M, M)
        attn_weights = self.dropout_layer(attn_weights) # (B, T, H, M, M)

        # Apply attention on V
        attn_output = attn_weights @ V # (B, T, H, M, d_k)

        # Concatenate heads
        attn_output = attn_output.transpose(2, 3).contiguous() # (B, T, M, H, d_k)
        attn_output = attn_output.view(batch_size, seq_len, units, self.d_model) # (B, T, M, D)

        # Final projection
        output = self.W_o(attn_output) # (B, T, M, D)

        return output
