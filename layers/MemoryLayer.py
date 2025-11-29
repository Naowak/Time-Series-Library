import torch
import numpy as np
from layers.ssm_triton import ssm_triton
from torch.utils.checkpoint import checkpoint

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

    def forward(self, X, state=None, chunk_size=256, use_checkpoint=False):
        """
        Forward pass of the reservoir network.
        
        Parameters:
        - X (torch.Tensor): Input tensor [batch, time, input_dim].
        - state (torch.Tensor, optional): Initial states [batch, units, neurons].
        - chunk_size (int): Chunk size for triton SSM computation.
        - use_checkpoint (bool): Whether to use gradient checkpointing.

        Returns:
        - new_state (torch.Tensor): Updated state [batch, time, units, neurons].
        - output (torch.Tensor): Output tensor [batch, time, units, output_dim].
        """  
        if use_checkpoint:
            output, new_state = checkpoint(self._forward_computation, X, state, chunk_size, use_reentrant=False)
        else:
            output, new_state = self._forward_computation(X, state, chunk_size)
        
        return output, new_state

    def _forward_computation(self, X, state=None, chunk_size=256): # Chunk réduit pour Fused (meilleure occupation)
        """
        Version Fused & Memory Optimized.
        """  
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        
        # 1. Projections
        X = X.view(batch_size, seq_len, 1, 1, self.input_dim) # [B, T, 1, 1, D]
        X_proj = torch.tanh(torch.einsum('btxyi,mio->btmyo', X, self.proj)) # [B, T, M, 1, D]

        if state is None:
            state = torch.zeros(batch_size, self.units, self.neurons, dtype=self.complex_dtype, device=self.device) # [B, M, R]
        
        # 2. Calcul du Leak Rate
        lr_proj = torch.einsum('btmxd,mdy->bmyt', X_proj, self.adaptive_lr) # [B, T, M, 1, D] @ [M, D, 1] -> [B, T, M, 1, 1] -> [B, M, 1, T]
        lr = torch.softmax(lr_proj / self.temperature, dim=1).contiguous() # [B, M, 1, T] / [M, 1, 1] -> [B, M, 1, T]  -> [B, M, 1, T]
        del lr_proj

        # 3. Calcul de U_pre
        u_pre = torch.einsum('btmxd,mdr->bmrt', X_proj.to(self.complex_dtype), self.Win_).contiguous() # [B, M, R, T]
        del X_proj 
        
        # 4. Préparation Lambda
        lam = self.Lambda.unsqueeze(0).expand(batch_size, -1, -1).contiguous() # [B, M, R]

        # 5. Préparation State
        h_init = state.contiguous() # [B, M, R]

        # 6. SSM Computation
        new_state = ssm_triton(u_pre, lr, lam, h_init, chunk_size=chunk_size) # [B, M, R, T]
        new_state = new_state.permute(0, 3, 1, 2) # [B, T, M, R]
        del u_pre, lr, lam, h_init

        # 7. Output Computation
        output = torch.einsum('btmr,mro->btmo', new_state, self.Wout_).real.to(self.dtype) # [B, T, M, D]
        output = torch.tanh(output)

        return output, new_state # [B, T, M, D], [B, T, M, R]

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

