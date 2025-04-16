import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import matplotlib.pyplot as plt
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
import math
import pickle

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class HalideGraphDataset(Dataset):
    """
    PyTorch Dataset for Halide execution time prediction using graph structure
    """
    def __init__(self, sequences, execution_times, seq_len=10, num_features=None):
        """
        Transforms sequence data into graph-structured data
        
        Args:
            sequences: numpy array of sequences
            execution_times: numpy array of execution times
            seq_len: length of each sequence
            num_features: number of features per node
        """
        self.sequences = sequences
        self.execution_times = torch.FloatTensor(execution_times).reshape(-1, 1)
        self.seq_len = seq_len
        self.num_features = num_features or sequences.shape[2]
        
        # Pre-process the data into graph format
        self.graph_data = [self._create_graph(seq) for seq in sequences]
        
    def _create_graph(self, sequence):
        """
        Creates a graph representation from a sequence
        
        For each element in the sequence:
        - Create a node with the element's features
        - Connect to the previous element (sequential edge)
        - Add skip connections to capture long-range dependencies
        """
        # Node features are directly from the sequence
        x = torch.FloatTensor(sequence)
        
        # Edge indices (source, target)
        edges = []
        
        # Sequential edges (connecting adjacent nodes)
        for i in range(self.seq_len - 1):
            edges.append((i, i + 1))
            edges.append((i + 1, i))  # Bidirectional
        
        # Skip connections (connect every node to nodes 2 and 3 steps away)
        for i in range(self.seq_len - 2):
            edges.append((i, i + 2))
            edges.append((i + 2, i))  # Bidirectional
            
        for i in range(self.seq_len - 3):
            edges.append((i, i + 3))
            edges.append((i + 3, i))  # Bidirectional
        
        # Full connectivity for the first and last nodes (to act as start/end points)
        for i in range(1, self.seq_len):
            edges.append((0, i))
            edges.append((i, 0))
            
        for i in range(self.seq_len - 1):
            edges.append((i, self.seq_len - 1))
            edges.append((self.seq_len - 1, i))
        
        # Convert to PyTorch tensor
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index)
        
        return data
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.graph_data[idx], self.execution_times[idx]


def load_dataset(file_path='halide_data.npz'):
    """
    Load the dataset created by the preprocessing script
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file {file_path} not found. Please run the preprocessing script first.")
    
    data = np.load(file_path)
    sequences = data['sequences']
    execution_times = data['execution_times']
    
    print(f"Loaded dataset with {len(sequences)} samples")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Execution times shape: {execution_times.shape}")
    
    return sequences, execution_times


# Define ScalerWrapper class to make the inverse transform function picklable
class ScalerWrapper:
    """
    Wrapper class for scaling and inverse scaling operations
    This makes the scaling operations picklable
    """
    def __init__(self, y_scaler):
        self.y_scaler = y_scaler
        
    def inverse_transform_y(self, y_scaled):
        """
        Apply inverse transform to convert scaled values back to original scale
        """
        y_log = self.y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
        y_original = np.expm1(y_log)  # Inverse of log1p
        return y_original
    
    def __call__(self, y_scaled):
        """
        Make the object callable for convenience
        """
        return self.inverse_transform_y(y_scaled)


def prepare_train_val_test_split(sequences, execution_times, test_size=20):
    """
    Split the dataset into training, validation and test sets with improved preprocessing.
    """
    # Get total number of samples
    n_samples = len(sequences)
    
    if n_samples <= test_size:
        raise ValueError(f"Not enough samples ({n_samples}) to create a test set of {test_size} samples")
    
    # Create indices and shuffle them
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Select test indices (exactly 20 samples)
    test_indices = indices[:test_size]
    remaining_indices = indices[test_size:]
    
    # Split the remaining data into training and validation (80/20 split)
    train_indices, val_indices = train_test_split(remaining_indices, test_size=0.2, random_state=42)
    
    # Feature scaling - standardize the sequence data
    x_scaler = StandardScaler()
    sequences_flat = sequences.reshape(-1, sequences.shape[2])
    sequences_scaled_flat = x_scaler.fit_transform(sequences_flat)
    sequences_scaled = sequences_scaled_flat.reshape(sequences.shape)
    
    # Apply RobustScaler or Log transformation to the execution times for better handling of outliers
    # Using log transformation for execution times as they can vary greatly in magnitude
    execution_times_log = np.log1p(execution_times)  # log1p to handle zeros
    
    # Then scale the log-transformed values
    y_scaler = MinMaxScaler()
    execution_times_scaled = y_scaler.fit_transform(execution_times_log.reshape(-1, 1)).flatten()
    
    # Print some statistics about the target variable
    print(f"Execution times statistics before scaling:")
    print(f"  Min: {np.min(execution_times)}, Max: {np.max(execution_times)}")
    print(f"  Mean: {np.mean(execution_times)}, Std: {np.std(execution_times)}")
    print(f"Execution times statistics after log transform and scaling:")
    print(f"  Min: {np.min(execution_times_scaled)}, Max: {np.max(execution_times_scaled)}")
    
    # Create graph dataset with scaled data
    # Extract sequence length from data shape
    seq_len = sequences.shape[1]
    num_features = sequences.shape[2]
    
    print(f"Creating graph dataset with sequence length {seq_len} and {num_features} features")
    full_dataset = HalideGraphDataset(sequences_scaled, execution_times_scaled, seq_len=seq_len, num_features=num_features)
    
    # Create dataset splits using indices
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    print(f"Split dataset into:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    # Create a picklable wrapper for the inverse transform function
    scaler_wrapper = ScalerWrapper(y_scaler)
    
    # Save the scaler for later use
    with open('y_scaler.pkl', 'wb') as f:
        pickle.dump(scaler_wrapper, f)
    
    return train_dataset, val_dataset, test_dataset, scaler_wrapper


class GraphAttentionCollator:
    """
    Custom collator for batching graph data
    """
    def __call__(self, batch):
        graphs = [item[0] for item in batch]
        targets = torch.stack([item[1] for item in batch])
        
        # Batch the graphs
        batched_graph = Batch.from_data_list(graphs)
        
        return batched_graph, targets


class SelfAttention(nn.Module):
    """
    Self-attention layer for processing node features
    """
    def __init__(self, in_features, out_features):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(in_features, out_features)
        self.key = nn.Linear(in_features, out_features)
        self.value = nn.Linear(in_features, out_features)
        self.scale = math.sqrt(out_features)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, in_features]
        q = self.query(x)  # [batch_size, seq_len, out_features]
        k = self.key(x)    # [batch_size, seq_len, out_features]
        v = self.value(x)  # [batch_size, seq_len, out_features]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [batch_size, seq_len, seq_len]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len, seq_len]
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, v)  # [batch_size, seq_len, out_features]
        
        return output


class GraphAttentionModel(nn.Module):
    """
    Graph Neural Network model with attention mechanism for execution time prediction
    """
    def __init__(self, num_node_features, hidden_dim=64, num_layers=3, dropout=0.2):
        super(GraphAttentionModel, self).__init__()
        
        # Multiple GNN layers with different attention heads
        self.conv1 = GATv2Conv(num_node_features, hidden_dim, heads=4, dropout=dropout, concat=True)
        expanded_dim = hidden_dim * 4  # Due to concat=True with 4 heads
        
        # Middle layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(expanded_dim, hidden_dim, heads=4, dropout=dropout, concat=True)
            )
        
        # Final GNN layer
        self.conv_final = GATv2Conv(expanded_dim, hidden_dim, heads=1, dropout=dropout, concat=False)
        
        # Batch normalization layers for stability
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(expanded_dim))
        self.batch_norm_final = nn.BatchNorm1d(hidden_dim)
        
        # Self-attention layer for capturing sequence patterns
        self.self_attention = SelfAttention(hidden_dim, hidden_dim)
        
        # Output layers with residual connections
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim // 4, 1)
        
        # Activation functions
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data):
        # Extract node features and edge indices
        x, edge_index = data.x, data.edge_index
        
        # First GNN layer
        x = self.conv1(x, edge_index)
        x = self.leaky_relu(x)
        x = self.batch_norms[0](x)
        x = self.dropout(x)
        
        # Middle GNN layers
        for i, conv in enumerate(self.convs):
            x_res = x  # Save for residual connection
            x = conv(x, edge_index)
            x = self.leaky_relu(x)
            x = self.batch_norms[i+1](x)
            x = self.dropout(x)
            # Add residual connection if dimensions match
            if x_res.shape[-1] == x.shape[-1]:
                x = x + x_res
        
        # Final GNN layer
        x = self.conv_final(x, edge_index)
        x = self.leaky_relu(x)
        x = self.batch_norm_final(x)
        
        # Reshape for self-attention
        batch_size = data.num_graphs
        nodes_per_graph = x.size(0) // batch_size
        x = x.view(batch_size, nodes_per_graph, -1)
        
        # Apply self-attention
        x = self.self_attention(x)
        
        # Global pooling: combine all node features for each graph
        x = x.mean(dim=1)  # Simple mean pooling across nodes
        
        # MLP for final prediction
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


def train_model(model, train_loader, val_loader, epochs=150, learning_rate=0.0005, patience=20, 
                weight_decay=1e-5, scheduler_factor=0.7, scheduler_patience=8):
    """
    Train the PyTorch Graph Neural Network model with improved training procedure
    """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True
    )
    
    # For early stopping
    best_val_loss = float('inf')
    counter = 0
    best_model = None
    
    # Training history
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    
    print(f"Starting training with learning rate: {learning_rate}")
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        
        for batch_data, targets in train_loader:
            batch_data = batch_data.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(batch_data)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * batch_data.num_graphs
            train_mae += torch.sum(torch.abs(outputs - targets)).item()
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        train_mae /= len(train_loader.dataset)
        train_maes.append(train_mae)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for batch_data, targets in val_loader:
                batch_data = batch_data.to(device)
                targets = targets.to(device)
                outputs = model(batch_data)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * batch_data.num_graphs
                val_mae += torch.sum(torch.abs(outputs - targets)).item()
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        val_mae /= len(val_loader.dataset)
        val_maes.append(val_mae)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Print stats
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{epochs}, LR: {current_lr:.6f}, Train Loss: {train_loss:.4f}, '
              f'Train MAE: {train_mae:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            counter = 0
            # Save best model
            torch.save(best_model, 'best_gnn_model.pth')
            print(f"Saved new best model with validation loss: {val_loss:.4f}")
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load the best model
    model.load_state_dict(best_model)
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    return model, {'train_loss': train_losses, 'val_loss': val_losses, 
                   'train_mae': train_maes, 'val_mae': val_maes}


def evaluate_model(model, test_loader, inverse_transform_y):
    """
    Evaluate the model on the test set and calculate error percentages
    with inverse transform to get original scale predictions
    """
    model.eval()
    
    y_true_scaled = []
    y_pred_scaled = []
    
    with torch.no_grad():
        for batch_data, targets in test_loader:
            batch_data = batch_data.to(device)
            targets = targets.to(device)
            outputs = model(batch_data)
            
            y_true_scaled.extend(targets.cpu().numpy().flatten())
            y_pred_scaled.extend(outputs.cpu().numpy().flatten())
    
    # Convert to numpy arrays
    y_true_scaled = np.array(y_true_scaled).reshape(-1, 1)
    y_pred_scaled = np.array(y_pred_scaled).reshape(-1, 1)
    
    # Apply inverse transform to get original scale
    y_true = inverse_transform_y(y_true_scaled)
    y_pred = inverse_transform_y(y_pred_scaled)
    
    # Calculate absolute errors
    absolute_errors = np.abs(y_pred - y_true)
    
    # Calculate percentage errors
    percentage_errors = (absolute_errors / np.clip(np.abs(y_true), 1e-10, None)) * 100
    
    # Calculate mean and standard deviation of errors
    mean_absolute_error = np.mean(absolute_errors)
    mean_percentage_error = np.mean(percentage_errors)
    median_percentage_error = np.median(percentage_errors)
    
    print("\nTest Set Evaluation:")
    print(f"Mean Absolute Error: {mean_absolute_error:.4f}")
    print(f"Mean Percentage Error: {mean_percentage_error:.2f}%")
    print(f"Median Percentage Error: {median_percentage_error:.2f}%")
    
    # Print individual predictions and errors
    print("\nIndividual Test Sample Results:")
    print("Sample | Actual Time | Predicted Time | Abs Error | Error %")
    print("-" * 70)
    
    for i in range(len(y_true)):
        print(f"{i:6d} | {y_true[i]:11.2f} | {y_pred[i]:14.2f} | {absolute_errors[i]:9.2f} | {percentage_errors[i]:7.2f}%")
    
    return y_pred, absolute_errors, percentage_errors, y_true


def plot_results(history, y_true, y_pred):
    """
    Plot training history and prediction results with improved visualizations
    """
    # Create a figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot training & validation loss with log scale
    ax1.semilogy(history['train_loss'], label='Train')
    ax1.semilogy(history['val_loss'], label='Validation')
    ax1.set_title('Model Loss (log scale)')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='upper right')
    ax1.grid(True, which="both", ls="-")
    
    # Plot training & validation MAE
    ax2.plot(history['train_mae'], label='Train')
    ax2.plot(history['val_mae'], label='Validation')
    ax2.set_title('Mean Absolute Error')
    ax2.set_ylabel('MAE')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    # Plot actual vs predicted values
    ax3.scatter(y_true, y_pred, alpha=0.8)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred)) * 1.1
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    ax3.set_title('Actual vs Predicted Execution Times')
    ax3.set_xlabel('Actual Time (ms)')
    ax3.set_ylabel('Predicted Time (ms)')
    ax3.legend()
    ax3.grid(True)
    
    # Add equal aspect ratio to make the comparison clearer
    ax3.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig('gnn_results.png', dpi=300)
    
    # Create additional plot for error analysis
    plt.figure(figsize=(12, 6))
    error_percentages = (np.abs(y_pred - y_true) / np.clip(np.abs(y_true), 1e-10, None)) * 100
    
    # Sort errors for better visualization
    sorted_indices = np.argsort(error_percentages)
    sorted_errors = error_percentages[sorted_indices]
    
    plt.bar(range(len(y_true)), sorted_errors)
    plt.axhline(np.mean(error_percentages), color='r', linestyle='--', 
                label=f'Mean Error: {np.mean(error_percentages):.2f}%')
    plt.axhline(np.median(error_percentages), color='g', linestyle='--', 
                label=f'Median Error: {np.median(error_percentages):.2f}%')
    plt.title('Percentage Error by Test Sample (Sorted)')
    plt.xlabel('Test Sample (Sorted by Error)')
    plt.ylabel('Error (%)')
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig('error_analysis_gnn.png', dpi=300)
    
    # Add histogram of errors
    plt.figure(figsize=(10, 6))
    plt.hist(error_percentages, bins=20, alpha=0.7, color='blue')
    plt.axvline(np.mean(error_percentages), color='r', linestyle='--', 
                label=f'Mean Error: {np.mean(error_percentages):.2f}%')
    plt.axvline(np.median(error_percentages), color='g', linestyle='--', 
                label=f'Median Error: {np.median(error_percentages):.2f}%')
    plt.title('Distribution of Percentage Errors')
    plt.xlabel('Error (%)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig('error_histogram_gnn.png', dpi=300)
    
    plt.show()


def main():
    # Load the dataset
    sequences, execution_times = load_dataset()
    
    # Split data into training, validation, and test sets with improved preprocessing
    train_dataset, val_dataset, test_dataset, scaler_wrapper = prepare_train_val_test_split(
        sequences, execution_times, test_size=20
    )
    
    # Use custom collator for graph data
    collator = GraphAttentionCollator()
    
    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collator)  # Use batch size 1 for detailed evaluation
    
    # Get sample input to determine feature size
    sample_batch = next(iter(train_loader))
    sample_data, _ = sample_batch
    num_node_features = sample_data.num_node_features
    print(f"Number of node features: {num_node_features}")
    
    # Initialize the GNN model
    model = GraphAttentionModel(
        num_node_features=num_node_features,
        hidden_dim=64,
        num_layers=3,
        dropout=0.2
    ).to(device)
    
    print(model)
    
    # Count the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    # Train the model with optimized hyperparameters
    model, history = train_model(
        model, 
        train_loader, 
        val_loader, 
        epochs=300,  # Increased epochs for better convergence
        learning_rate=0.0003,  # Lower learning rate for stability
        patience=25,  # More patience for early stopping
        weight_decay=1e-5,  # L2 regularization
        scheduler_factor=0.7,
        scheduler_patience=10
    )
    
    # Evaluate the model on the test set
    y_pred, absolute_errors, percentage_errors, y_true = evaluate_model(model, test_loader, scaler_wrapper)
    
    # Plot results
    plot_results(history, y_true, y_pred)
    
    # Save final model - now saving only what's needed
    torch.save({
        'model_state_dict': model.state_dict(),
        # Not saving the scaler_wrapper here as it's already saved to disk
        'model_class_name': 'GraphAttentionModel',
        'num_node_features': num_node_features,
        'hidden_dim': 64,
        'num_layers': 3,
        'dropout': 0.2
    }, 'final_gnn_model.pth')
    
    print("\nModel training and evaluation complete!")
    print("Full model saved to 'final_gnn_model.pth'")
    print("Scaler saved to 'y_scaler.pkl'")


# Function to load model and make predictions (for future use)
def load_model_and_predict(model_path='final_gnn_model.pth', scaler_path='y_scaler.pkl'):
    """
    Load a saved model and its scaler to make predictions
    """
    # Load model parameters
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model instance
    model = GraphAttentionModel(
        num_node_features=checkpoint['num_node_features'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint['dropout']
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler_wrapper = pickle.load(f)
    
    return model, scaler_wrapper


if __name__ == "__main__":
    main()
