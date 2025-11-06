import torch
from layers.MemoryLayer import MemoryLayer
from layers.AttentionLayer import AttentionLayer
from layers.FeedForwardLayer import FeedForwardLayer
import pickle

class DynamicalTransformer(torch.nn.Module):
    def __init__(self, num_layers=1, memory_units=4, memory_dim=64, attention_dim=16, attention_heads=4, dropout=0.0, 
                 memory_connectivity=0.1, device='cpu', dtype=torch.float32, complex_dtype=torch.complex64):
        """Initialize the DynamicalTransformer module."""
        super(DynamicalTransformer, self).__init__()

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
        self.input_dim = None
        self.output_dim = None
        self.learning_rate = None
        self.weight_decay = None
        self.classification = None

        # Initialize layers
        self.input_projection = None # [I, D]
        self.output_projection = None # [M*D, O]
        self.layers = torch.nn.ModuleList()

        # Optimizer & Loss
        self.optimizer = None
        self.criterion = None
        
        # Dropout & Norm
        self.dropout_layer = torch.nn.Dropout(dropout)
        self.norm = torch.nn.RMSNorm(attention_dim, eps=1e-8, device=device, dtype=dtype)

    def forward(self, x, state=None, return_state=False):
        """
        Args:
            x: Input tensor of shape (B, T, I) where B is batch size, T is sequence length, I is input dimension
            state: Optional state tensor for encoder-decoder attention of shape (B, T, M, R)
        Returns:
            output: Tensor of shape (B, T, O) where O is output dimension
        """
        # Input projection
        x = torch.tensor(x, dtype=self.dtype, device=self.device)
        step = self.input_projection(x) # (B, T, D)

        # Pass through layers
        for layer in self.layers:

            # Memory Layer
            if isinstance(layer, MemoryLayer):
                memory, state = layer(step, state=state) # (B, T, M, D), (B, T, M, R)
                step = memory # (B, T, M, D)
            
            # Attention Layer
            elif isinstance(layer, AttentionLayer):
                step = step + layer(step, state=memory) # (B, T, M, D)
                step = self.norm(self.dropout_layer(step)) # (B, T, M, D)

            # Feed-Forward Layer
            elif isinstance(layer, FeedForwardLayer):
                step = step + layer(step) # (B, T, M, D)
                step = self.norm(self.dropout_layer(step)) # (B, T, M, D)

            # Unknown Layer
            else:
                raise ValueError("Unknown layer type in DynamicalTransformer.")

        # Output projection
        step = step.reshape(step.shape[0], step.shape[1], -1) # (B, T, M*D)
        output = self.output_projection(step) # (B, T, O)

        if return_state:
            return output, state
        return output

    def run_training(self, X_train, Y_train, T_train, X_valid=None, Y_valid=None, T_valid=None,
                     epochs=100, batch_size=32, learning_rate=1e-3, weight_decay=1e-5, classification=False,
                     patience=5, min_delta=1e-5, path=None):
        """Run the training loop for the model."""
        # Convert data to tensors
        X_train = torch.tensor(X_train, dtype=self.dtype, device=self.device)
        Y_train = torch.tensor(Y_train, dtype=self.dtype, device=self.device)
        T_train = torch.tensor(T_train, device=self.device)

        if X_valid is not None and Y_valid is not None and T_valid is not None:
            X_valid = torch.tensor(X_valid, dtype=self.dtype, device=self.device)
            Y_valid = torch.tensor(Y_valid, dtype=self.dtype, device=self.device)
            T_valid = torch.tensor(T_valid, device=self.device)

        # Define model if not already defined
        if self.input_dim is None or self.output_dim is None:
            self._define_model(
                input_dim=X_train.shape[-1],
                output_dim=Y_train.shape[-1],
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                classification=classification
            )
        
        # Create DataLoader
        train_dataset = torch.utils.data.TensorDataset(X_train, Y_train, T_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Loop variables
        history = {'train_loss': [], 'val_loss': []}
        patience_counter = 0
        best_val_loss = float('inf')

        # Training loop
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0

            # For each batch
            for X_batch, Y_batch, T_batch in train_loader:
                # Reset gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.forward(X_batch) # (B, T, O)
                
                # Compute loss & backpropagate
                loss = self._compute_loss(X_batch, Y_batch, outputs, T_batch, classification)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)

            # Compute average loss & store
            epoch_loss /= len(train_loader.dataset)
            history['train_loss'].append(epoch_loss)

            # Validation
            if X_valid is not None and Y_valid is not None and T_valid is not None:
                self.eval()
                with torch.no_grad():
                    # Forward pass
                    val_outputs = self.forward(X_valid) # (B, T, O)

                    # Compute validation loss
                    val_loss = self._compute_loss(X_valid, Y_valid, val_outputs, T_valid, classification)
                    history['val_loss'].append(val_loss.item())
                
                # Early stopping check
                if val_loss.item() < best_val_loss - min_delta:
                    best_val_loss = val_loss.item()
                    patience_counter = 0
                    # Save model
                    if path is not None:
                        self.save(path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Stop {epoch+1}, Best val loss: {best_val_loss:.4f}, Path: {path}")
                        break

            # Print progress
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}", end='')
            if X_valid is not None and Y_valid is not None and T_valid is not None:
                print(f", Val Loss: {val_loss.item():.4f}")
            else:
                print()


    def run_inference(self, X, batch_size=32, state=None, return_state=False):
        """Run inference on the model."""
        # Convert data to tensor
        X = torch.tensor(X, dtype=self.dtype, device=self.device)

        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(X)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Inference loop
        self.eval()
        outputs_list = []
        states_list = []
        with torch.no_grad():
            for (X_batch,) in data_loader:
                # Forward pass
                if return_state:
                    outputs_batch, state = self.forward(X_batch, state=state, return_state=True)
                    states_list.append(state)
                else:
                    outputs_batch = self.forward(X_batch, state=state)

                outputs_list.append(outputs_batch.cpu())

        # Concatenate outputs
        outputs = torch.cat(outputs_list, dim=0) # (N, T, O)
        states = torch.cat(states_list, dim=0) if return_state else None

        if return_state:
            return outputs.numpy(), states.numpy()
        return outputs.numpy()
    
    def run_autoregressive(self, X, num_steps, batch_size=10):
        """Run autoregressive inference on the model."""
        # Convert data to tensor
        X = torch.tensor(X, dtype=self.dtype, device=self.device)
        state = None # /!\ Initialize state to None, in future we can allow passing an initial state

        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(X)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Autoregressive inference loop
        self.eval()
        outputs_list = []
        with torch.no_grad():
            for (X_batch,) in data_loader:

                inputs = X_batch # (B, 1, I)
                outputs_batch = []

                for _ in range(num_steps):
                    # Forward pass
                    output_step, state = self.forward(inputs, state=state, return_state=True) # (B, T, O), (B, T, M, R)
                    state = state[:, -1, :, :] # (B, M, R)

                    # Store last output & prepare next input
                    inputs = output_step[:, -1:, :] # (B, 1, I)
                    outputs_batch.append(inputs) # (B, 1, O)

                # Concatenate outputs for the batch
                outputs_batch = torch.cat(outputs_batch, dim=1) # (B, num_steps, O)
                outputs_list.append(outputs_batch.cpu())

        # Concatenate all outputs
        outputs = torch.cat(outputs_list, dim=0) # (N, num_steps, O)
        outputs = torch.cat([X.cpu(), outputs], dim=1) # (N, T + num_steps, O)
        return outputs.numpy()

    def _define_model(self, input_dim, output_dim, learning_rate, weight_decay, classification=False):
        """Define the model architecture, optimizer, and loss function."""
        # Store hyper-parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.classification = classification

        # Input & Output projection
        self.input_projection = torch.nn.Linear(input_dim, self.attention_dim, device=self.device, dtype=self.dtype)
        self.output_projection = torch.nn.Linear(self.memory_units * self.attention_dim, output_dim, device=self.device, dtype=self.dtype)

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

        # Create Memory-Decoder Attention Layer
        for i in range(self.num_layers):
            attention_layer = AttentionLayer(
                d_model=self.attention_dim,
                num_heads=self.attention_heads,
                dropout=self.dropout,
                self_attention=i==0, # First layer is self-attention
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

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Loss function
        if classification:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()

        print(f"Model defined with {self._count_params()} parameters.")

    
    def save(self, path):
        """Save the model state to the specified path."""
        pickle.dump(self, open(path, 'wb'))

    @staticmethod
    def load(path):
        model = pickle.load(open(path, 'rb'))
        return model

    def _count_params(self):
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters()) # if p.requires_grad

    def _compute_loss(self, X, Y, outputs, T, classification):
        """
        Calcule la perte.

        Paramètres :
        - X : Données d'entrée (batch_size, seq_len, input_dim).
        - Y : Données de sortie (batch_size, seq_len, output_dim).
        - outputs : Sorties du modèle (batch_size, seq_len, output_dim).
        - T : Indices des pas de temps pour les prédictions (batch_size, seq_len).
        - classification (bool) : Indique si la tâche est une classification.
        """
        # Select only the prediction timesteps
        preds = []
        truths = []
        for j in range(X.shape[0]):
            preds += [outputs[j, T[j], :]]
            truths += [Y[j, T[j], :]]
        preds = torch.stack(preds)
        truths = torch.stack(truths)

        if classification:
            # Prepare truth tensor for CrossEntropyLoss
            truths = torch.argmax(truths, dim=-1).view(-1) # [B * prediction_timesteps] int: class
            preds = preds.view(-1, Y.shape[-1]) # [B * prediction_timesteps, O] float: logits

        # Compute loss
        loss = self.criterion(preds, truths)
        return loss



































