import torch
import numpy as np


class MemoryLayer(torch.nn.Module):
    """Implements a reservoir network."""

    def __init__(self, units=None, neurons=None, input_dim=None, output_dim=None, res_connectivity=0.2, 
                 input_connectivity=0.2, device='cpu', dtype=torch.float32, complex_dtype=torch.complex64):
        """
        Create a reservoir with the given parameters.

        Args:
        - units (int): Number of reservoirs.
        - neurons (int): Number of neurons in each reservoir.
        - input_dim (int): Input dimension.
        - output_dim (int): Output dimension.
        - input_scaling (float): Input scaling.
        - res_connectivity (float): Connectivity of the recurrent weight matrix.
        - input_connectivity (float): Connectivity of the input weight matrix.
        - bias_prob (float): Probability of bias.
        - activation (str): Activation function to use ('id', 'relu', 'gelu', 'tanh', 'sigmoid', 'softmax') after the reservoir.
        - device (str): Device to use ('cpu' or 'cuda').
        - dtype (torch.dtype): Data type of the tensors.
        """
        super().__init__()
        # Check the parameters
        if units is None or neurons is None or input_dim is None or output_dim is None:
            raise ValueError("You must provide the number of units, neurons and input/output dimension")
        
        # Store the parameters
        self.units = units # M
        self.neurons = neurons # R
        self.input_dim = input_dim # D
        self.output_dim = output_dim # D
        self.res_connectivity = res_connectivity
        self.input_connectivity = input_connectivity
        self.device = device
        self.dtype = dtype
        self.complex_dtype = complex_dtype

        # Create matrices
        proj = _initialize_matrix((units, input_dim, input_dim), 1.0, distribution='normal', dtype=dtype, device=device)
        adaptive_lr = _initialize_matrix((units, input_dim, 1), 1.0, distribution='normal', dtype=dtype, device=device)
        Win = _initialize_matrix((units, input_dim, neurons), input_connectivity, distribution='bernoulli', dtype=dtype, device=device)
        W = _initialize_matrix((units, neurons, neurons), res_connectivity, distribution='normal', dtype=dtype, device=device)
        sr = _initialize_matrix((units, 1, 1), 1.0, distribution='uniform', dtype=dtype, device=device)
        Wout = _initialize_matrix((units, neurons, output_dim), 1.0, distribution='normal', dtype=dtype, device=device)

        # Handle Spectral Radius
        initial_sr = _get_spectral_radius(W).view(units, 1, 1)
        W = W / initial_sr * sr # Init SR

        # Compute Lambda, P, P_inv for the spectral decomposition of W
        W = W.to(torch.float64)  # High precision for stability
        Lambda, P, P_inv = list(zip(*[_decompose_matrix(w) for w in W]))
        Lambda = torch.stack([torch.diagonal(lam) for lam in Lambda]).to(complex_dtype) # [M, R]
        P = torch.stack(P).to(complex_dtype) # [M, R, R]
        P_inv = torch.stack(P_inv).to(complex_dtype) # [M, R, R]

        # Register proj, Win_ & Lambda 
        self.proj = torch.nn.Parameter(proj) # [M, D, D]
        self.adaptive_lr = torch.nn.Parameter(adaptive_lr) # [M, D, 1]
        self.temperature = torch.nn.Parameter(torch.ones(units, 1, 1, dtype=dtype, device=device)) # [M, 1, 1]
        self.Win_ = torch.nn.Buffer(Win.to(complex_dtype) @ P) # [M, D, R]
        self.Lambda = torch.nn.Parameter(Lambda) # [M, R]
        self.Wout_ = torch.nn.Parameter(P_inv @ Wout.to(complex_dtype)) # [M, R, D]
        # self.rmsnorm = torch.nn.RMSNorm(normalized_shape=output_dim, eps=1e-8)

    def forward(self, X, state=None):
        """
        Forward pass of the reservoir network.
        
        Args:
        - X (torch.Tensor): Input tensor [batch, time, input_dim].
        - state (torch.Tensor, optional): Initial states [batch, units, neurons].

        Returns:
        - new_state (torch.Tensor): Updated state [batch, time, units, neurons].
        - output (torch.Tensor): Output tensor [batch, time, units, output_dim].
        """  
        # Extract constants from X and reshape it
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        X = X.view(batch_size, seq_len, 1, 1, self.input_dim) # [B, T, 1, 1, D]
        X_proj = torch.tanh(X @ self.proj) # [B, T, 1, 1, D] @ [M, D, D] = [B, T, M, 1, D]

        # Reshape the state, if not provided, initialize it
        if state is None:
            state = torch.zeros(batch_size, 1, self.units, 1, self.neurons, dtype=self.complex_dtype, device=self.device) # [B, 1, M, 1, R]
        else:
            state = state.view(batch_size, 1, self.units, 1, self.neurons).to(self.complex_dtype) # [B, 1, M, 1, R]

        # Compute lr with adaptive leak rate
        lr_logits = X_proj @ self.adaptive_lr / self.temperature # [B, T, M, 1, 1]
        lr = torch.softmax(lr_logits, dim=2) # [B, T, M, 1, 1]
        Win_ = lr * self.Win_ # [B, T, M, 1, 1] * [M, D, R] = [B, T, M, D, R]
        Lambda = self.Lambda.view(1, 1, self.units, 1, self.neurons) # [1, 1, M, 1, R]
        Lambda = lr * Lambda + (1 - lr) # [B, T, M, 1, 1] * [1, 1, M, 1, R] = [B, T, M, 1, R]
        
        # Compute Lambda window
        # Lambda_window = _compute_Lambda_window(Lambda.squeeze(3)) # [B, T, T+1, M, R]

        # Compute feed and concat the initial state
        feed = X_proj.to(self.complex_dtype) @ Win_ # [B, T, M, 1, R]
        feed = torch.cat((state, feed), dim=1) # [B, T+1, M, 1, R]
        feed = feed.view(batch_size, 1, seq_len+1, self.units, self.neurons) # [B, 1, T+1, M, R]

        # Compute echos, states and apply RMSNorm
        # echos = (feed * Lambda_window) # [B, T, T+1, M, R]
        # new_state = torch.sum(echos, dim=2) # [B, T, M, R] -> Sum over the echo dimension
        Lambda_compact = _compute_Lambda_compact(Lambda.squeeze(3)) # [B, (T^2+T)/2, M, R]
        feed = _compute_feed_compact(feed.squeeze(1)) # [B, (T^2+T)/2, M, R]
        echos = (feed * Lambda_compact) # [B, (T^2+T)/2, M, R]
        new_state = _compute_new_state(echos, seq_len) # [B, T, M, R]

        # Compute the output
        output = (new_state.unsqueeze(3) @ self.Wout_).real.to(self.dtype).squeeze(3) # [B, T, M, D]
        output = torch.tanh(output.to(self.dtype)) # Apply activation function [B, T, M, D]
        # output = self.rmsnorm(output) # Apply RMSNorm [B, T, M, D]

        return output, new_state  # [B, T, M, D], [B, T, M, R]
    
    def __repr__(self):
        txt = "MemoryLayer("
        txt += f"\n  (proj): Tensor{tuple(self.proj.shape)},"
        txt += f"\n  (adaptive_lr): Tensor{tuple(self.adaptive_lr.shape)},"
        txt += f"\n  (temperature): Tensor{tuple(self.temperature.shape)},"
        txt += f"\n  (Win): Tensor{tuple(self.Win_.shape)},"
        txt += f"\n  (Lambda): Tensor{tuple(self.Lambda.shape)},"
        txt += f"\n  (Wout): Tensor{tuple(self.Wout_.shape)}"
        txt += "\n)"
        return txt


def _initialize_matrix(shape, connectivity, distribution='normal', dtype=torch.float32, device='cpu'):
    """
    Initialize a matrix with a given shape and connectivity.

    Args:
    - shape (tuple): Shape of the matrix.
    - connectivity (float): Connectivity of the matrix.
    - distribution (str): Distribution of the matrix values ('normal' or 'bernoulli').
    - kwargs: Additional arguments for the distribution.

    Returns:
    - torch.Tensor: Initialized matrix.
    """
    if distribution == 'normal':
        matrix = torch.tensor(np.random.normal(size=shape, loc=0, scale=1), device=device, dtype=dtype)
        mask = _fixed_bernoulli(shape, connectivity, device=device, dtype=dtype)
        return matrix * mask
    
    elif distribution == 'uniform':
        matrix = torch.rand(size=shape, device=device, dtype=dtype)
        mask = _fixed_bernoulli(shape, connectivity, device=device, dtype=dtype)
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

    Args:
    - matrix (torch.Tensor): The matrix to analyze.

    Returns:
    - float: The spectral radius of the matrix.
    """
    # Convert the matrix to float64 for better precision
    matrix = matrix.to(torch.float64) # eigenvalues does not support dtype

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

    Args:
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
    matrix = torch.zeros(shape, device=device)
    
    # For each column, set the connections
    for head in range(shape[-3]):
        for col in range(shape[-1]):
            indices = torch.randperm(shape[-2])[:nb_connections]
            matrix[head, indices, col] = 1
    
    return matrix.to(dtype)

def _decompose_matrix(W):
    """
    Décompose la matrice W en valeurs propres et vecteurs propres.
    
    Args:
        W (torch.Tensor): Matrice à décomposer.
        
    Returns:
        Lambda (torch.Tensor): Matrice diagonale des valeurs propres.
        P (torch.Tensor): Matrice des vecteurs propres.
        P_inv (torch.Tensor): Inverse de la matrice des vecteurs propres.
    """
    # Construction de la décomposition en valeurs propres
    eigenvalues, eigenvectors = torch.linalg.eig(W.to(torch.float64)) # High precision for stability
    Lambda = torch.diag(eigenvalues)
    P = eigenvectors
    P_inv = torch.linalg.inv(P)

    return Lambda, P, P_inv


def _compute_Lambda_window(Lambda):
    """
    Compute the windowed lambda matrices for all time steps.

    Args:
        Lambda: Tensor of shape (B, T, M, R) representing eigenvalues (R) combined with adaptive leak rates for each time step.
    Returns:
        Lambda_window: Tensor of shape (B, T, T+1, M, R) where Lambda_window[b, t, k] = product of Lambda[b, j] for j=k to t-1
                       and Lambda_window[b, t, t+1] = 1 (identity for the product when k=t).
    """
    B, T, M, R = Lambda.shape
    Lambda_window = torch.zeros((B, T, T+1, M, R), dtype=Lambda.dtype, device=Lambda.device)

    for t in range(T):
        # Compute cumulative products for each batch and memory unit
        Lambda_window[:, t, :t+1] = torch.cumprod(Lambda[:, :t + 1].flip(1), dim=1).flip(1)
        Lambda_window[:, t, t+1] = 1  # Identity for the product when k=t

    return Lambda_window


def _compute_Lambda_compact(Lambda):
    """
    Compute the compact representation of Lambda for efficient computation.

    Args:
        Lambda: Tensor of shape (B, T, M, R) representing eigenvalues (R) combined with adaptive leak rates for each time step.

    Returns:
        Lambda_compact: Tensor of shape (B, (T^2+T)/2, M, R) representing the compacted version of Lambda.
    """
    B, T, M, R = Lambda.shape
    Lambda_compact = torch.zeros((B, (T**2 + T) // 2 + T, M, R), dtype=Lambda.dtype, device=Lambda.device)

    # Fill the compact representation
    idx = 0
    for t in range(1, T+1):
        Lambda_compact[:, idx:idx + t] = torch.cumprod(Lambda[:, :t].flip(1), dim=1).flip(1)
        Lambda_compact[:, idx + t] = 1  # Identity for the product when k=t
        idx += t + 1

    return Lambda_compact

def _compute_feed_compact(feed):
    """
    Apply the compacted Lambda to the feed tensor.

    Args:
        feed: Tensor of shape (B, T+1, M, R) representing the feed inputs.

    Returns:
        new_state: Tensor of shape (B, T, M, R) representing the new states after applying Lambda.
    """
    B, T_plus_1, M, R = feed.shape
    T = T_plus_1 - 1
    feed_compact = torch.zeros((B, (T**2 + T) // 2 + T, M, R), dtype=feed.dtype, device=feed.device)

    idx = 0
    for t in range(1, T+1):
        feed_compact[:, idx:idx + t+1] = feed[:, :t+1]
        idx += t + 1

    return feed_compact

def _compute_new_state(echos, T):
    """
    Compute the new state from the echos tensor.
    Args:
        echos: Tensor of shape (B, (T^2+T)/2, M, R) representing the echos.
    Returns:
        new_state: Tensor of shape (B, T, M, R) representing the new states.
    """
    B, _, M, R = echos.shape
    new_state = torch.zeros((B, T, M, R), dtype=echos.dtype, device=echos.device)

    idx = 0
    for t in range(T):
        new_state[:, t] = torch.sum(echos[:, idx:idx + t + 2], dim=1)
        idx += t + 2

    return new_state
