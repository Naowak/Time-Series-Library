import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
from layers.MemoryLayer import MemoryLayer
from layers.AttentionLayer import AttentionLayer
from layers.FeedForwardLayer import FeedForwardLayer


class Model(nn.Module):
    """
    Dynamical Transformer (DT) model adapted for Time Series Library interface
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # === Configuration Setup ===
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.num_class = getattr(configs, 'num_class', 1)  # For classification tasks
        
        # DT specific parameters (with defaults if not in configs)
        self.num_layers = getattr(configs, 'e_layers', 1)  # Use e_layers from TSL convention
        self.memory_units = getattr(configs, 'memory_units', 4)
        self.memory_dim = getattr(configs, 'memory_dim', 64)
        self.attention_dim = getattr(configs, 'd_model', 64)  # Use d_model from TSL convention
        self.attention_heads = getattr(configs, 'n_heads', 4)  # Use n_heads from TSL convention
        self.dropout_rate = getattr(configs, 'dropout', 0.1)
        self.memory_connectivity = getattr(configs, 'memory_connectivity', 0.1)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float32
        self.complex_dtype = torch.complex64
        
        # === Model Components ===
        # Input embedding layer (converts raw input to model dimension)
        self.embedding = DataEmbedding(
            configs.enc_in, self.attention_dim, configs.embed, configs.freq, self.dropout_rate
        )
        
        # Core DynamicalTransformer
        self.dynamical_transformer = DynamicalTransformerCore(
            num_layers=self.num_layers,
            memory_units=self.memory_units,
            memory_dim=self.memory_dim,
            attention_dim=self.attention_dim,
            attention_heads=self.attention_heads,
            dropout=self.dropout_rate,
            memory_connectivity=self.memory_connectivity,
            device=self.device,
            dtype=self.dtype,
            complex_dtype=self.complex_dtype
        )
        
        # === Task-specific Output Layers ===
        if self.task_name == 'classification':
            # Classification needs special handling: flatten sequence then classify
            self.act = F.gelu
            self.dropout = nn.Dropout(self.dropout_rate)
            self.projection = nn.Linear(self.memory_units * self.attention_dim * self.seq_len, configs.num_class)
        else:
            # All other tasks: direct projection from attention_dim to output_dim
            self.projection = nn.Linear(self.memory_units * self.attention_dim, configs.c_out, bias=True)
    
    def _dt_forward_pass(self, x, normalize=False):
        """
        Core DT forward pass - shared by all tasks
        
        Args:
            x: Input tensor [B, L, D] 
            normalize: Whether to apply normalization (for forecasting)
            
        Returns:
            x_out: Processed tensor [B, L, D]
            normalization_params: (mean, std) if normalize=True, else (None, None)
        """
        batch_size, seq_len, _ = x.shape
        mean_enc, std_enc = None, None
        
        # === Normalization (only for forecasting) ===
        if normalize:
            mean_enc = x.mean(1, keepdim=True).detach()  # B x 1 x E
            x = x - mean_enc
            std_enc = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / std_enc
        
        # === Embedding ===
        x = self.embedding(x, None)  # [B, L, D]
        
        # === Pass through DynamicalTransformer ===
        x_out = self.dynamical_transformer(x)  # [B, L, D]
        
        return x_out, (mean_enc, std_enc)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Main forward method following TSL interface
        
        Args:
            x_enc: Encoder input [B, L, D]
            x_mark_enc: Encoder temporal features [B, L, D] (unused in DT)
            x_dec: Decoder input [B, L, D] (unused in DT)
            x_mark_dec: Decoder temporal features [B, L, D] (unused in DT)
            mask: Mask for imputation tasks (unused in current implementation)
            
        Returns:
            Model output based on task_name
        """
        
        # === Forecasting Tasks ===
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # Normalize input for forecasting stability
            x_out, (mean_enc, std_enc) = self._dt_forward_pass(x_enc, normalize=True) # [B, L, M*D]
            
            # Project to output dimension
            dec_out = self.projection(x_out)  # [B, L, C]
            
            # Denormalize output
            if mean_enc is not None and std_enc is not None:
                dec_out = dec_out * (std_enc[:, :, :self.c_out].repeat(1, dec_out.shape[1], 1))
                dec_out = dec_out + (mean_enc[:, :, :self.c_out].repeat(1, dec_out.shape[1], 1))
            
            # Return only prediction horizon
            return dec_out[:, -self.pred_len:, :]  # [B, pred_len, C]
        
        elif self.task_name in ['imputation', 'anomaly_detection']:
            # No normalization for imputation/anomaly detection
            x_out, _ = self._dt_forward_pass(x_enc, normalize=False) # [B, L, M*D]
            
            # Project to output dimension
            dec_out = self.projection(x_out)  # [B, L, C]
            
            return dec_out  # [B, L, C]
        
        elif self.task_name == 'classification':
            # No normalization for classification
            x_out, _ = self._dt_forward_pass(x_enc, normalize=False) # [B, L, M*D]
            
            # Apply activation and dropout
            output = self.act(x_out)  # [B, L, M*D]
            output = self.dropout(output)
            
            # Zero-out padding embeddings if available
            if x_mark_enc is not None:
                output = output.reshape(output.shape[0], output.shape[1], self.memory_units, -1) # [B, L, M, D]
                output = output * x_mark_enc.unsqueeze(-1) # [B, L, M, D]
                output = output.reshape(output.shape[0], output.shape[1], -1) # [B, L, M*D]
            
            # Flatten and classify
            output = output.reshape(output.shape[0], -1)  # [B, L*M*D]
            output = self.projection(output)  # [B, num_classes]
            
            return output
        
        raise ValueError(f"Unsupported task: {self.task_name}")


class DynamicalTransformerCore(nn.Module):
    """
    Core DynamicalTransformer implementation that can be used as a building block
    """
    
    def __init__(self, num_layers=1, memory_units=4, memory_dim=64, attention_dim=16, 
                 attention_heads=4, dropout=0.0, memory_connectivity=0.1, 
                 device='cpu', dtype=torch.float32, complex_dtype=torch.complex64):
        """Initialize the DynamicalTransformer core module."""
        super(DynamicalTransformerCore, self).__init__()

        # Store hyper-parameters
        self.num_layers = num_layers
        self.memory_units = memory_units
        self.memory_dim = memory_dim
        self.attention_dim = attention_dim
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.memory_connectivity = memory_connectivity
        self.device = device
        self.dtype = dtype
        self.complex_dtype = complex_dtype

        # Initialize layers
        self.layers = nn.ModuleList()
        
        # Dropout & Norm
        self.dropout_layer = nn.Dropout(dropout)
        self.norm = nn.RMSNorm(attention_dim, eps=1e-8, device=device, dtype=dtype)
        
        # Build model architecture
        self._build_model()

    def _build_model(self):
        """Build the model architecture."""
        
        # Create Memory Layer
        memory_layer = MemoryLayer(
            units=self.memory_units,
            neurons=self.memory_dim,
            input_dim=self.attention_dim,
            output_dim=self.attention_dim,
            input_connectivity=self.memory_connectivity,
            res_connectivity=self.memory_connectivity,
            device=self.device,
            dtype=self.dtype,
            complex_dtype=self.complex_dtype
        )
        self.layers.append(memory_layer)

        # Create Memory-Decoder Attention and Feed-Forward Layers
        for i in range(self.num_layers):
            attention_layer = AttentionLayer(
                d_model=self.attention_dim,
                num_heads=self.attention_heads,
                dropout=self.dropout,
                self_attention=i==0,  # First layer is self-attention
                dtype=self.dtype,
                device=self.device
            )
            feedforward_layer = FeedForwardLayer(
                d_model=self.attention_dim,
                d_ff=4*self.attention_dim,
                dropout=self.dropout,
                device=self.device,
                dtype=self.dtype
            )
            self.layers.append(attention_layer)
            self.layers.append(feedforward_layer)

    def forward(self, x, state=None, return_state=False):
        """
        Args:
            x: Input tensor of shape (B, T, D) where B is batch size, T is sequence length, D is input dimension
            state: Optional state tensor for encoder-decoder attention of shape (B, T, M, R)
        Returns:
            output: Tensor of shape (B, T, D) where D is output dimension
        """
        # Ensure correct device and dtype
        x = x.to(device=self.device, dtype=self.dtype)
        step = x  # (B, T, D)
        
        # Initialize memory variable
        memory = None

        # Pass through layers
        for layer in self.layers:

            # Memory Layer
            if isinstance(layer, MemoryLayer):
                memory, state = layer(step, state=state)  # (B, T, M, D), (B, T, M, R)
                step = memory  # (B, T, M, D)
            
            # Attention Layer
            elif isinstance(layer, AttentionLayer):
                step = step + layer(step, state=memory)  # (B, T, M, D)
                step = self.norm(self.dropout_layer(step))  # (B, T, M, D)

            # Feed-Forward Layer
            elif isinstance(layer, FeedForwardLayer):
                step = step + layer(step)  # (B, T, M, D)
                step = self.norm(self.dropout_layer(step))  # (B, T, M, D)

            # Unknown Layer
            else:
                raise ValueError("Unknown layer type in DynamicalTransformer.")

        # # Output is the mean over memory units
        # if len(step.shape) == 4:  # (B, T, M, D)
        #     output = step.mean(dim=2)  # (B, T, D)
        # else:
        #     output = step
        output = step.reshape(step.shape[0], step.shape[1], -1)  # (B, T, M*D)

        if return_state:
            return output, state
        return output
