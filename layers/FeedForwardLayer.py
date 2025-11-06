import torch


class FeedForwardLayer(torch.nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, dtype=torch.float32, device='cpu'):
        """
        Initialize the FeedForward module.

        Args:
            d_model: Model dimension (D)
            d_ff: Feed-forward dimension (usually 4 * D)
            dropout: Dropout rate
        """
        super().__init__()
        # Store hyper-parameters
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Linear layers for the feed-forward network
        self.linear1 = torch.nn.Linear(d_model, d_ff, device=device, dtype=dtype)  # [D, D_ff]
        self.linear2 = torch.nn.Linear(d_ff, d_model, device=device, dtype=dtype)  # [D_ff, D]

        # Activation function
        self.activation = torch.nn.GELU()
            
        # Dropout layer
        self.dropout_layer = torch.nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, M, D) where B is batch size, T is sequence length, M is number of units, D is model dimension

        Returns:
            output: Tensor of shape (B, T, M, D)
        """
        # Apply first linear layer + activation
        x = self.linear1(x)  # (B, T, M, D_ff)
        x = self.activation(x) # (B, T, M, D_ff)

        # Apply dropout
        x = self.dropout_layer(x) # (B, T, M, D_ff)

        # Apply second linear layer
        x = self.linear2(x)  # (B, T, M, D)

        return x