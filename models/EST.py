import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
import numpy as np

class Model(nn.Module):
    """
    Echo State Transformer (EST) model adapted for Time Series Library interface
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # === Configuration Setup ===
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        
        # EST specific parameters (with defaults if not in configs)
        self.num_layers = getattr(configs, 'num_layers', 1)
        self.memory_units = getattr(configs, 'memory_units', 4)
        self.memory_dim = getattr(configs, 'memory_dim', 100)
        self.attention_dim = getattr(configs, 'd_model', 64)
        self.dropout_rate = getattr(configs, 'dropout', 0.1)
        self.memory_connectivity = getattr(configs, 'memory_connectivity', 0.2)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float32
        
        # === Model Components ===
        # Input embedding layer (converts raw input to model dimension)
        self.embedding = DataEmbedding(
            configs.enc_in, self.attention_dim, configs.embed, configs.freq, self.dropout_rate
        )
        
        # Dropout for regularization
        self.fc_in_dropout = nn.Dropout(self.dropout_rate)
        
        # Stack of EST layers (core of the model)
        self.est_layers = nn.ModuleList([
            ESTLayer(
                self.memory_units, self.memory_dim, self.attention_dim, 
                self.dropout_rate, self.memory_connectivity, self.device, self.dtype
            ) for _ in range(self.num_layers)
        ])
        
        # === Task-specific Output Layers ===
        if self.task_name == 'classification':
            # Classification needs special handling: flatten sequence then classify
            self.act = F.gelu
            self.dropout = nn.Dropout(self.dropout_rate)
            self.projection = nn.Linear(self.attention_dim * self.seq_len, configs.num_class)
        else:
            # All other tasks: direct projection from attention_dim to output_dim
            self.projection = nn.Linear(self.attention_dim, configs.c_out, bias=True)
            
        # Initialize EST layers
        for layer in self.est_layers:
            layer._define_model()
    
    def _est_forward_pass(self, x, normalize=False, steps=0):
        """
        Core EST forward pass - shared by all tasks
        
        Args:
            x: Input tensor [B, L, D] 
            normalize: Whether to apply normalization (for forecasting)
            steps: Number of autoregressive prediction steps (0 = no autoregression)
            
        Returns:
            x_out: Processed tensor [B, L+steps, D]
            normalization_params: (mean, std) if normalize=True, else (None, None)
        """
        batch_size, seq_len, _ = x.shape
        mean_enc, std_enc = None, None
        
        # === Normalization (only for forecasting) ===
        if normalize:
            mean_enc = x.mean(1, keepdim=True).detach()
            x = x - mean_enc
            std_enc = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x = x / std_enc
        
        # === Embedding ===
        x = self.embedding(x, None)  # [B, L, D]
        
        # === Initialize EST States ===
        states = torch.zeros(
            batch_size, self.num_layers, self.memory_units, self.memory_dim,
            dtype=self.dtype, device=x.device
        )
        
        # === Sequential Processing through EST Layers ===
        outputs = []
        current_x = x
        
        # Process original sequence
        for t in range(seq_len):
            # Get current timestep
            x_t = self.fc_in_dropout(current_x[:, t, :])  # [B, D]
            
            # Process through all EST layers
            new_states = []
            for i, layer in enumerate(self.est_layers):
                x_t, new_state = layer(x_t, states[:, i])
                new_states.append(new_state)
            
            # Update states for next timestep
            states = torch.stack(new_states, dim=1)
            outputs.append(x_t)
        
        # === Autoregressive Prediction ===
        if steps > 0:
            for step in range(steps):
                # Use last output as input for next prediction
                last_output = outputs[-1]  # [B, D]
                
                # Project to get next input (if needed)
                next_input = self.projection(last_output)  # [B, c_out]
                
                # Re-embed the prediction for next step
                if next_input.shape[-1] != self.attention_dim:
                    # Need to re-embed if output dim != attention dim
                    next_input_expanded = next_input.unsqueeze(1)  # [B, 1, c_out]
                    next_embedded = self.embedding(next_input_expanded, None)  # [B, 1, D]
                    next_x_t = self.fc_in_dropout(next_embedded.squeeze(1))  # [B, D]
                else:
                    next_x_t = self.fc_in_dropout(next_input)  # [B, D]
                
                # Process through all EST layers
                new_states = []
                for i, layer in enumerate(self.est_layers):
                    next_x_t, new_state = layer(next_x_t, states[:, i])
                    new_states.append(new_state)
                
                # Update states
                states = torch.stack(new_states, dim=1)
                outputs.append(next_x_t)
        
        # === Reconstruct Sequence ===
        x_out = torch.stack(outputs, dim=1)  # [B, L+steps, D]
        
        return x_out, (mean_enc, std_enc)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, steps=0):
        """
        Main forward method following TSL interface
        
        Args:
            x_enc: Encoder input [B, L, D]
            x_mark_enc: Encoder temporal features [B, L, D] 
            x_dec: Decoder input [B, L, D] (unused in EST)
            x_mark_dec: Decoder temporal features [B, L, D] (unused in EST)
            mask: Mask for imputation tasks (unused in current implementation)
            steps: Number of autoregressive steps for forecasting
            
        Returns:
            Model output based on task_name
        """
        
        # === Forecasting Tasks ===
        if self.task_name in ['long_term_forecast']:
            # Use autoregressive steps for forecasting
            forecast_steps = self.pred_len - self.seq_len
            
            # Process with normalization for stable training
            x_out, (mean_enc, std_enc) = self._est_forward_pass(x_enc, normalize=True, steps=forecast_steps)
            # x_out, _ = self._est_forward_pass(x_enc, normalize=False, steps=forecast_steps)
            
            # Project to output dimension
            x_out = self.projection(x_out)
            
            # Denormalize predictions
            if mean_enc is not None and std_enc is not None:
                x_out = x_out * std_enc + mean_enc
            
            # Return only prediction horizon
            return x_out  # [B, pred_len, D]
        
        elif self.task_name in ['short_term_forecast']:
            # Process without normalization (preserve original scale)
            x_out, (mean_enc, std_enc) = self._est_forward_pass(x_enc, normalize=False, steps=0)
            
            # Project to output dimension
            x_out = self.projection(x_out)

            # Denormalize predictions
            if mean_enc is not None and std_enc is not None:
                x_out = x_out * std_enc + mean_enc
            
            return x_out[:, -self.pred_len:, :]  # [B, pred_len, D]
        
        # === Imputation & Anomaly Detection ===
        elif self.task_name in ['imputation', 'anomaly_detection']:
            # Process without normalization (preserve original scale)
            x_out, _ = self._est_forward_pass(x_enc, normalize=False, steps=0)
            
            # Project to output dimension
            x_out = self.projection(x_out)
            
            return x_out  # [B, L, D]
        
        # === Classification ===
        elif self.task_name == 'classification':
            # Process sequence
            x_out, _ = self._est_forward_pass(x_enc, normalize=False, steps=0)
            
            # Apply activation and dropout
            x_out = self.act(x_out)
            x_out = self.dropout(x_out)
            
            # Mask padding tokens if temporal features provided
            if x_mark_enc is not None:
                x_out = x_out * x_mark_enc.unsqueeze(-1)
            
            # Flatten sequence for classification
            x_out = x_out.reshape(x_out.shape[0], -1)  # [B, L*D]
            
            # Final classification layer
            x_out = self.projection(x_out)  # [B, num_classes]
            
            return x_out
        
        raise ValueError(f"Unsupported task: {self.task_name}")


class ESTLayer(torch.nn.Module):
    """Implementation of the Echo State Transformer layer."""

    def __init__(self, memory_units=4, memory_dim=100, attention_dim=3, dropout=0, memory_connectivity=0.2, device='cpu', dtype=torch.float32):
        """
        Initialize the Echo State Transformer layer.

        Parameters:
        - memory_units (int): Number of memory units (M).
        - memory_dim (int): Dimension of each memory unit (R).
        - attention_dim (int): Dimension of the attention mechanism (D).
        - dropout (float): Dropout rate.
        - memory_connectivity (float): Connectivity of the memory units.
        - device (str): Device to use ('cpu' or 'cuda').
        - dtype (torch.dtype): Data type of the tensors.
        """
        super(ESTLayer, self).__init__()
        # Store model parameters
        self.memory_units = memory_units # M
        self.memory_dim = memory_dim # R
        self.attention_dim = attention_dim # D
        self.dropout = dropout
        self.memory_connectivity = memory_connectivity
        self.device = device
        self.dtype = dtype

        # Norm, activation, dropout
        self.norm1 = torch.nn.RMSNorm(normalized_shape=(attention_dim), eps=1e-8, device=device, dtype=self.dtype)
        self.norm2 = torch.nn.RMSNorm(normalized_shape=(attention_dim), eps=1e-8, device=device, dtype=self.dtype)
        self.norm3 = torch.nn.RMSNorm(normalized_shape=(attention_dim), eps=1e-8, device=device, dtype=self.dtype)
        self.gelu = torch.nn.GELU()
        self.attn_dropout = torch.nn.Dropout(self.dropout)
        self.self_attn_dropout = torch.nn.Dropout(self.dropout)
        self.ff_out_dropout = torch.nn.Dropout(dropout)

        # Initialize memory with fixed weights
        self.memory = None
        
        # Query, Key, Value weights 
        self.Wq = None # [M, D, D]
        self.Wk = None # [M, D, D]
        self.Wv = None # [M, D, D]
        self.SWq = None # [D, D]
        self.SWk = None # [D, D]
        self.SWv = None # [D, D]

        # Other Layers
        self.Wreduce = None # [M*D, D]
        self.ff_in = None # [D, 4*D]
        self.ff_out = None # [4*D, D]

    def forward(self, emb, Si):
        """
        Forward pass of an Echo State Transformer layer.

        Parameters:
        - emb (torch.Tensor): Input tensor [B, I]
        - Si (torch.Tensor, optional): Reservoirs states [B, M, R].

        Returns:
        - Y (torch.Tensor): Output tensor [B, O]
        """
        # Init batch_size & previous state
        batch_size = emb.shape[0] # B
        Si_ = Si # [B, M, R]
        Sout_ = (Si_.unsqueeze(2) @ self.memory.Wout).squeeze(2) # [B, M, D] # /!\ Can be computed once

        # Attention on previous states
        Q = emb.view(batch_size, 1, 1, self.attention_dim) @ self.Wq # [B, M, 1, D]
        K = Sout_.unsqueeze(1) @ self.Wk.unsqueeze(0) # [B, M, M, D]
        V = Sout_.unsqueeze(1) @ self.Wv.unsqueeze(0) # [B, M, M, D]
        attn = torch.nn.functional.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout if self.training else 0) # [B, M, 1, D]
        update = self.norm1(self.attn_dropout(attn.squeeze(2)) + emb.unsqueeze(1)) # [B, M, D]

        # Memory update
        Si, Sout = self.memory(update, Si_) # [B, M, R], [B, M, D]

        # Attention on current state
        SQ = Sout @ self.SWq # [B, M, D]
        SK = Sout @ self.SWk # [B, M, D]
        SV = Sout @ self.SWv # [B, M, D]
        self_attn = torch.nn.functional.scaled_dot_product_attention(SQ, SK, SV, dropout_p=self.dropout if self.training else 0) # [B, M, D]
        self_update = self.norm2(self.self_attn_dropout(self_attn) + Sout) # [B, M, D]

        # Knowledge Enhancement
        SUreduce = self.Wreduce(self_update.view(batch_size, self.memory_units * self.attention_dim)) # [B, D]
        Z = self.gelu(self.ff_in(SUreduce))
        OUT = self.norm3(self.ff_out_dropout(self.ff_out(Z)) + SUreduce)

        return OUT, Si # [B, O], [B, M, R]

    def _define_model(self):
        """
        Define the model architecture.

        Parameters:
        - input_dim (int): Input dimension.
        - output_dim (int): Output dimension.
        - learning_rate (float): Learning rate for the optimizer.
        - weight_decay (float): Weight decay for the optimizer.
        - classification (bool): Indicates if the task is a classification task.
        """
        # Initialize memory with fixed weights
        self.memory = Memory(units=self.memory_units, neurons=self.memory_dim, input_dim=self.attention_dim, output_dim=self.attention_dim, 
                            res_connectivity=self.memory_connectivity, input_connectivity=self.memory_connectivity,
                             device=self.device, dtype=self.dtype)
        
        # Query, Key, Value weights
        self.Wq = torch.nn.Parameter(torch.empty(self.memory_units, self.attention_dim, self.attention_dim, dtype=self.dtype, device=self.device)) # [M, D, D]
        self.Wk = torch.nn.Parameter(torch.empty(self.memory_units, self.attention_dim, self.attention_dim, dtype=self.dtype, device=self.device)) # [M, D, D]
        self.Wv = torch.nn.Parameter(torch.empty(self.memory_units, self.attention_dim, self.attention_dim, dtype=self.dtype, device=self.device)) # [M, D, D]
        self.SWq = torch.nn.Parameter(torch.empty(self.attention_dim, self.attention_dim, dtype=self.dtype, device=self.device)) # [D, D]
        self.SWk = torch.nn.Parameter(torch.empty(self.attention_dim, self.attention_dim, dtype=self.dtype, device=self.device)) # [D, D]
        self.SWv = torch.nn.Parameter(torch.empty(self.attention_dim, self.attention_dim, dtype=self.dtype, device=self.device)) # [D, D]
        torch.nn.init.kaiming_uniform_(self.Wq, a=5**0.5)
        torch.nn.init.kaiming_uniform_(self.Wk, a=5**0.5)
        torch.nn.init.kaiming_uniform_(self.Wv, a=5**0.5)
        torch.nn.init.kaiming_uniform_(self.SWq, a=5**0.5)
        torch.nn.init.kaiming_uniform_(self.SWk, a=5**0.5)
        torch.nn.init.kaiming_uniform_(self.SWv, a=5**0.5)

        # Linear layers (kaiming uniform init)
        self.Wreduce = torch.nn.Linear(self.memory_units * self.attention_dim, self.attention_dim, bias=True, dtype=self.dtype, device=self.device) # [M*D, D]
        self.ff_in = torch.nn.Linear(self.attention_dim, 4 * self.attention_dim, bias=True, dtype=self.dtype, device=self.device) # [D, 4*D]
        self.ff_out = torch.nn.Linear(4 * self.attention_dim, self.attention_dim, bias=True, dtype=self.dtype, device=self.device) # [4*D, D]


class Memory(torch.nn.Module):
    """Implements a reservoir network."""

    def __init__(self, units=None, neurons=None, input_dim=None, output_dim=None, input_scaling=1.0, res_connectivity=0.2, 
                 input_connectivity=0.2, bias_prob=0.5, device='cpu', dtype=torch.float32):
        """
        Create a reservoir with the given parameters.

        Parameters:
        - units (int): Number of reservoirs.
        - neurons (int): Number of neurons in each reservoir.
        - input_dim (int): Input dimension.
        - output_dim (int): Output dimension.
        - input_scaling (float): Input scaling.
        - res_connectivity (float): Connectivity of the recurrent weight matrix.
        - input_connectivity (float): Connectivity of the input weight matrix.
        - bias_prob (float): Probability of bias.
        - device (str): Device to use ('cpu' or 'cuda').
        - dtype (torch.dtype): Data type of the tensors.
        """
        super(Memory, self).__init__()
        # Check the parameters
        if units is None or neurons is None or input_dim is None or output_dim is None:
            raise ValueError("You must provide the number of units, neurons and input/output dimension")
        
        # Store the parameters
        self.units = units # M
        self.neurons = neurons # R
        self.input_dim = input_dim # D
        self.output_dim = output_dim # D
        self.input_scaling = input_scaling
        self.res_connectivity = res_connectivity
        self.input_connectivity = input_connectivity
        self.bias_prob = bias_prob
        self.device = device
        self.dtype = dtype

        # Create matrices
        W = _initialize_matrix((units, neurons, neurons), res_connectivity, distribution='normal', dtype=dtype, device=device)
        Win = _initialize_matrix((units, input_dim, neurons), input_connectivity, distribution='fixed_bernoulli', dtype=dtype, device=device)
        bias = _initialize_matrix((units, 1, neurons), bias_prob, distribution='bernoulli', dtype=dtype, device=device)
        Wout = _initialize_matrix((units, neurons, output_dim), 1.0, distribution='normal', dtype=dtype, device=device)
        adaptive_lr = torch.nn.init.uniform_(torch.empty((units, input_dim, 1), device=device, dtype=dtype))
        initial_sr = _get_spectral_radius(W).view(units, 1, 1)
        sr = torch.rand(units, 1, 1, dtype=dtype, device=device)
        temperature = torch.ones(1, dtype=dtype, device=device)

        # Register W, Win & bias as buffer
        self.register_buffer('W', W / initial_sr) # [M, R, R] Set SR to 1
        self.register_buffer('Win', Win) # [M, D, R]
        self.register_buffer('bias', bias) # [M, 1, R]
        #self.W = torch.nn.Buffer(W / initial_sr) # [M, R, R] Make W a parameter
        #self.Win = torch.nn.Buffer(Win) # [M, D, R]
        #self.bias = torch.nn.Buffer(bias) # [M, 1, R]

        # Register parameters 
        self.Wout = torch.nn.Parameter(Wout) # [M, R, D]
        self.sr = torch.nn.Parameter(sr) # [M, 1, 1]
        self.adaptive_lr = torch.nn.Parameter(adaptive_lr) # [M, D, 1] 
        self.temperature = torch.nn.Parameter(temperature) # [1]

        # Register the non-zero positions of W and Win, and the corresponding positions in x
        self.w_pos = W.transpose(-2, -1).nonzero(as_tuple=True) 
        self.win_pos = Win.transpose(-2, -1).nonzero(as_tuple=True)
        self.xw_pos = (self.w_pos[0], torch.zeros(self.w_pos[1].shape, dtype=int, device=device), self.w_pos[2])
        self.xwin_pos = (self.win_pos[0], torch.zeros(self.win_pos[1].shape, dtype=int, device=device), self.win_pos[2])
        

    def forward(self, X, state):
        """
        Forward pass of the reservoir network.
        
        Parameters:
        - X (torch.Tensor): Input tensor [batch, units, input_dim].
        - state (torch.Tensor, optional): Initial states [batch, units, neurons].

        Returns:
        - new_state (torch.Tensor): Updated state [batch, units, neurons].
        """  
        # Reshape X & state
        batch_size = X.shape[0] # B
        X = X.view(batch_size, self.units, 1, self.input_dim) # [B, M, 1, D]
        state = state.view(batch_size, self.units, 1, self.neurons) # [B, M, 1, R]

        # Adaptive Leak Rate
        lr = torch.softmax((X @ self.adaptive_lr) / self.temperature, dim=1) # [batch, units, 1, 1]

        # Feed
        feed = _sparse_mm_subhead(X, self.Win,  self.xwin_pos, self.win_pos, None) # [B, subM=k, 1, R]
        # feed = _sparse_mm_subhead(X, self.Win,  self.xwin_pos, self.win_pos, w_heads) # [B, subM=k, 1, R]

        # Adjust the spectral radius
        W = self.W * self.sr # [M, R, R]

        # Echo
        echo = _sparse_mm_subhead(state, W, self.xw_pos, self.w_pos, None) + self.bias # [B, subM=k, 1, R]
        # echo = _sparse_mm_subhead(state, W, self.xw_pos, self.w_pos, w_heads) + self.bias[w_heads] # [B, subM=k, 1, R]

        # Update the selected heads
        new_state = ((1 - lr) * state) + lr * torch.tanh(feed + echo) # [B, subM=k, 1, R]
        # heads_updated = ((1 - lr[batchs, w_heads]) * state[batchs, w_heads]) + lr[batchs, w_heads] * torch.tanh(feed + echo) # [B, subM=k, 1, R]
        # new_state = state.clone() # [B, M, 1, R]
        # new_state[batchs, w_heads] = heads_updated # [B, M, 1, R]

        output = new_state @ self.Wout # [B, M, 1, D]

        return new_state.squeeze(2), output.squeeze(2) # [B, M, R], [B, M, D] 
    
    def clamp_hp(self):
        """
        Clamp the hyperparameters of the reservoir.
        """
        # self.lr.data.clamp_(1e-5, 1)
        self.sr.data.clamp_(1e-5, 10)
    


def _initialize_matrix(shape, connectivity, distribution='normal', dtype=torch.float32, device='cpu'):
    """
    Initialize a matrix with a given shape and connectivity.

    Parameters:
    - shape (tuple): Shape of the matrix.
    - connectivity (float): Connectivity of the matrix.
    - distribution (str): Distribution of the matrix values ('normal' or 'bernoulli').
    - kwargs: Additional arguments for the distribution.

    Returns:
    - torch.Tensor: Initialized matrix.
    """
    if distribution == 'normal':
        matrix = torch.tensor(np.random.normal(size=shape, loc=0, scale=1), device=device, dtype=dtype)
        mask = _fixed_bernoulli(shape, connectivity, device=device)
        return matrix * mask
    
    elif distribution == 'bernoulli':
        return torch.bernoulli(torch.full(shape, connectivity, device=device, dtype=dtype))
    
    elif distribution == 'fixed_bernoulli':
        return _fixed_bernoulli(shape, connectivity, device=device)
    
    else:
        raise ValueError("Unsupported distribution type")

def _get_spectral_radius(matrix):
    """
    Get the spectral radius of a matrix.

    Parameters:
    - matrix (torch.Tensor): The matrix to analyze.

    Returns:
    - float: The spectral radius of the matrix.
    """
    # Convert the matrix to float32
    matrix = matrix.to(torch.float32) # eigenvalues does not support dtype

    # Compute the eigenvalues
    device = str(matrix.device)
    if 'mps' in device:
        # MPS does not support eigvals
        eigenvalues = torch.linalg.eigvals(matrix.to('cpu')).to(matrix.device) # So we temporarily move the matrix to the CPU
    else:
        eigenvalues = torch.linalg.eigvals(matrix) 
    
    # Compute the maximum eigenvalue
    abs_eigenvalue = torch.sqrt(eigenvalues.real**2 + eigenvalues.imag**2) # Cuda does not support torch.abs on Complex Number
    spectral_radius = torch.max(abs_eigenvalue, dim=-1).values

    return spectral_radius

def _fixed_bernoulli(shape, connectivity, device='cpu', dtype=torch.float32):
    """
    Generate a connectivity matrix with a given shape and connectivity.

    Parameters:
    - shape (tuple): Shape of the matrix (head, line, column) or (line, column).
    - connectivity (float): Connectivity of the matrix.

    Every column has the same number of ones. (This constraint allows sparse matrix multiplication)

    Returns:
    - torch.Tensor: Connectivity matrix (head, line, column).
    """

    # Check the connectivity
    if not 0 < connectivity <= 1:
        raise ValueError("Connectivity must be > 0 et <= 1")
    
    # If len(shape) == 2, add a dimension
    if len(shape) == 2:
        shape = (1, shape[0], shape[1])
    
    # Init matrix & nb connections
    nb_connections = max(1, int(connectivity * shape[-2])) # At least one connection
    matrix = torch.zeros(shape, device=device, dtype=dtype)
    
    # For each column, set the connections
    for head in range(shape[-3]):
        for col in range(shape[-1]):
            indices = torch.randperm(shape[-2])[:nb_connections]
            matrix[head, indices, col] = 1
    
    return matrix

def _sparse_mm_subhead(x, W, x_pos, w_pos, w_heads):
    """
    Perform a sparse matrix multiplication between x and W on a subset of W heads for each batch of x.

    Parameters:
    - x (torch.Tensor): Input tensor [B, H, 1, D].
    - W (torch.Tensor): Weight tensor [B, H, D, O].
    - x_pos (torch.Tensor): Indices of the corresponding values in x.
    - w_pos (torch.Tensor): Indices of the non-zero values in W.
    - w_heads (int): Heads to use per batch. [B, subH]

    With B as the batch size, H as the number of heads, O as the number of output dim, and D as the dim of the input sequence.

    Returns:
    - torch.Tensor: Result tensor [B, subH, 1, O].
    """
    # Extract subW (only non-zero values) & select heads
    subW = W.transpose(-2, -1)[w_pos].reshape(W.shape[0], W.shape[-1], -1).transpose(-2, -1) # [H, connectivity, O]
    # subW = subW[w_heads] # [B, subH, connectivity, O]

    # Extract subX (corresponding values of x) & select heads per batch
    subX = x[:, *x_pos].reshape(x.shape[0], x.shape[1], W.shape[-1], -1).transpose(-2, -1) # [B, H, connectivity, O]
    # batchs = torch.arange(x.shape[0]).view(-1, 1).expand_as(w_heads) # [B, subH]
    # subX = subX[batchs, w_heads] # [B, subH, connectivity, O]

    # Compute the result 
    result = (subW * subX).sum(-2).view(subX.shape[0], subX.shape[1], 1, subW.shape[-1]) # [B, subH, 1, O]

    return result











