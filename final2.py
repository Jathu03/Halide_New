import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import random
from sklearn.preprocessing import RobustScaler
import pickle
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.data import Data, Batch
from collections import defaultdict
from tqdm import tqdm
import time

# Set random seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

set_seed(42)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class AdvancedScalerWrapper:
    """
    Enhanced wrapper for scaling execution times.
    """
    def __init__(self, scaler_type='log_robust'):
        self.scaler_type = scaler_type
        self.scaler = None
        self.offset = 0

    def fit_transform(self, y):
        min_y = np.min(y)
        if min_y <= 0:
            self.offset = abs(min_y) + 1.0
        y_log = np.log1p(y + self.offset)
        self.scaler = RobustScaler()
        y_scaled = self.scaler.fit_transform(y_log.reshape(-1, 1)).flatten()
        return y_scaled

    def transform(self, y):
        y_log = np.log1p(y + self.offset)
        return self.scaler.transform(y_log.reshape(-1, 1)).flatten()

    def inverse_transform_y(self, y_scaled):
        y_log = self.scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
        return np.expm1(y_log) - self.offset

    def __call__(self, y_scaled):
        return self.inverse_transform_y(y_scaled)

class HalideGraphDataset(Dataset):
    """
    Dataset for Halide execution time prediction using graph structure.
    """
    def __init__(self, sequences, execution_times, seq_len=44, num_features=None):
        self.sequences = sequences
        self.execution_times = torch.FloatTensor(execution_times).reshape(-1, 1)
        self.seq_len = seq_len
        self.num_features = num_features or sequences.shape[2]
        
        self.graph_data = []
        self.adj_lists = []
        for i, seq in enumerate(sequences):
            graph, adj_list = self._create_graph(seq, i)
            self.graph_data.append(graph)
            self.adj_lists.append(adj_list)
        
    def _create_graph(self, sequence, idx):
        x = torch.FloatTensor(sequence)
        
        edges = []
        adj_list = defaultdict(list)
        for i in range(self.seq_len - 1):
            edges.append((i, i + 1))
            edges.append((i + 1, i))
            adj_list[i].append(i + 1)
            adj_list[i + 1].append(i)
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        data = Data(x=x, edge_index=edge_index, graph_idx=idx)
        return data, adj_list
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.graph_data[idx], self.execution_times[idx], self.adj_lists[idx]

class GraphAttentionCollator:
    """
    Custom collator for batching graph data and adjacency lists.
    """
    def __call__(self, batch):
        graphs = [item[0] for item in batch]
        targets = torch.stack([item[1] for item in batch])
        adj_lists = [item[2] for item in batch]
        batched_graph = Batch.from_data_list(graphs)
        return batched_graph, targets, adj_lists

class GraphLSTMCell(nn.Module):
    """
    Graph-LSTM cell that incorporates neighbor information.
    """
    def __init__(self, input_dim, hidden_dim):
        super(GraphLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        expected_input = input_dim + hidden_dim
        self.expected_input = expected_input
        print(f"GraphLSTMCell: input_dim={input_dim}, hidden_dim={hidden_dim}, expected_input={expected_input}")
        
        self.W_i = nn.Linear(expected_input, hidden_dim)
        self.W_f = nn.Linear(expected_input, hidden_dim)
        self.W_c = nn.Linear(expected_input, hidden_dim)
        self.W_o = nn.Linear(expected_input, hidden_dim)
        self.W_n = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, h_prev, c_prev, neighbors_h_batch):
        assert x.shape[-1] == self.input_dim, f"Expected x dim {self.input_dim}, got {x.shape[-1]}"
        assert h_prev.shape[-1] == self.hidden_dim, f"Expected h_prev dim {self.hidden_dim}, got {h_prev.shape[-1]}"
        
        combined = torch.cat([x, h_prev], dim=-1)
        assert combined.shape[-1] == self.expected_input, f"Expected combined dim {self.expected_input}, got {combined.shape[-1]}"
        
        i = torch.sigmoid(self.W_i(combined))
        f = torch.sigmoid(self.W_f(combined))
        c_tilde = torch.tanh(self.W_c(combined))
        o = torch.sigmoid(self.W_o(combined))
        
        # Aggregate neighbor hidden states for the batch
        if neighbors_h_batch is not None and any(len(nh) > 0 for nh in neighbors_h_batch):
            # Initialize neighbor contributions
            neighbor_contrib = torch.zeros_like(h_prev)
            for b in range(len(neighbors_h_batch)):
                neighbors_h = neighbors_h_batch[b]
                if len(neighbors_h) > 0:
                    neighbor_h = torch.mean(torch.stack(neighbors_h), dim=0)
                    neighbor_contrib[b] = self.W_n(neighbor_h)
            c_tilde = c_tilde + neighbor_contrib
        else:
            # No neighbors, no contribution
            pass
        
        c = f * c_prev + i * c_tilde
        h = o * torch.tanh(c)
        
        return h, c

class GraphLSTM(nn.Module):
    """
    Graph-LSTM model for execution time prediction.
    """
    def __init__(self, input_dim, seq_len, hidden_dim=256, num_layers=2, dropout=0.3):
        super(GraphLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        print(f"GraphLSTM: initializing with input_dim={input_dim}, hidden_dim={hidden_dim}")
        
        self.cells = nn.ModuleList([
            GraphLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data, adj_lists):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        
        batch_size = data.num_graphs
        h = [torch.zeros(batch_size, self.seq_len, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.seq_len, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        
        outputs = []
        h_updates = [[] for _ in range(self.num_layers)]
        c_updates = [[] for _ in range(self.num_layers)]
        
        for t in range(self.seq_len):
            node_indices = torch.arange(t, x.size(0), self.seq_len, device=x.device)
            node_features = x[node_indices]  # Shape: [batch_size, input_dim]
            
            layer_input = node_features
            for layer in range(self.num_layers):
                h_prev = h[layer][:, t, :]  # Shape: [batch_size, hidden_dim]
                c_prev = c[layer][:, t, :]  # Shape: [batch_size, hidden_dim]
                
                # Collect neighbor hidden states for the batch
                neighbors_h_batch = []
                for b in range(batch_size):
                    neighbors = adj_lists[b].get(t, [])
                    neighbors_h = [
                        h[layer][b, n % self.seq_len]
                        for n in neighbors
                        if (n < self.seq_len) and (n >= 0)
                    ]
                    neighbors_h_batch.append(neighbors_h)
                
                # Process batch
                h_t, c_t = self.cells[layer](
                    layer_input, 
                    h_prev, 
                    c_prev, 
                    neighbors_h_batch
                )
                
                h_updates[layer].append(h_t)
                c_updates[layer].append(c_t)
                layer_input = h_t
            
            outputs.append(h_updates[-1][-1])
        
        for layer in range(self.num_layers):
            h[layer] = torch.stack(h_updates[layer], dim=1)
            c[layer] = torch.stack(c_updates[layer], dim=1)
        
        final_h = torch.mean(torch.stack(outputs), dim=0)
        final_h = self.dropout(final_h)
        out = self.fc(final_h)
        return out.squeeze(-1)

def load_dataset(file_path='halide_data.npz'):
    """
    Load the dataset.
    """
    data = np.load(file_path)
    sequences = data['sequences'].astype(np.float32)
    execution_times = data['execution_times'].astype(np.float32)
    print(f"Loaded dataset with {len(sequences)} samples")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Execution times shape: {execution_times.shape}")
    return sequences, execution_times

def prepare_train_val_test_split(sequences, execution_times, test_size=20, val_size=0.2, 
                                 scaler_type='log_robust', feature_scaling='robust'):
    """
    Split the dataset with preprocessing.
    """
    n_samples = len(sequences)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    test_indices = indices[:test_size]
    remaining_indices = indices[test_size:]
    train_indices, val_indices = train_test_split(
        remaining_indices, test_size=val_size, random_state=42
    )
    
    x_scaler = RobustScaler()
    sequences_flat = sequences[train_indices].reshape(-1, sequences.shape[2])
    x_scaler.fit(sequences_flat)
    sequences_scaled = x_scaler.transform(sequences.reshape(-1, sequences.shape[2])).reshape(sequences.shape)
    
    y_scaler = AdvancedScalerWrapper(scaler_type=scaler_type)
    y_scaler.fit_transform(execution_times[train_indices])
    execution_times_scaled = y_scaler.transform(execution_times)
    
    seq_len = sequences.shape[1]
    num_features = sequences.shape[2]
    
    full_dataset = HalideGraphDataset(
        sequences_scaled, execution_times_scaled, seq_len=seq_len, num_features=num_features
    )
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    print(f"Split dataset into:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    with open('y_scaler_advanced.pkl', 'wb') as f:
        pickle.dump(y_scaler, f)
    with open('x_scaler.pkl', 'wb') as f:
        pickle.dump(x_scaler, f)
    
    return train_dataset, val_dataset, test_dataset, y_scaler, seq_len

def train_model_with_fold(model, train_loader, val_loader, fold=0, 
                          epochs=90, learning_rate=0.001, min_lr=1e-6,
                          weight_decay=1e-5, patience=10,
                          gradient_accumulation_steps=1,
                          use_amp=True):
    """
    Train a single model with improved training strategy.
    """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    scaler = GradScaler() if use_amp else None
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = f'best_graph_lstm_model_fold_{fold}.pth'
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        train_count = 0
        optimizer.zero_grad()
        
        for i, (data, target, adj_lists) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            data = data.to(device)
            target = target.squeeze(-1).to(device)
            
            with autocast(enabled=use_amp):
                output = model(data, adj_lists)
                loss = criterion(output, target) / gradient_accumulation_steps
            if use_amp:
                scaler.scale(loss).backward()
                if (i + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                loss.backward()
                if (i + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
            
            train_loss += loss.item() * gradient_accumulation_steps * data.num_graphs
            train_count += data.num_graphs
        
        train_loss /= train_count
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_count = 0
        with torch.no_grad():
            for data, target, adj_lists in val_loader:
                data = data.to(device)
                target = target.squeeze(-1).to(device)
                with autocast(enabled=use_amp):
                    output = model(data, adj_lists)
                    loss = criterion(output, target)
                val_loss += loss.item() * data.num_graphs
                val_mae += torch.abs(output - target).sum().item()
                val_count += data.num_graphs
        
        val_loss /= val_count
        val_mae /= val_count
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Time: {time.time() - epoch_start_time:.2f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    model.load_state_dict(torch.load(best_model_path))
    os.remove(best_model_path)
    return train_losses, val_losses

def evaluate_model(model, test_loader, device, y_scaler):
    """
    Evaluate model and compute error percentages.
    """
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data, target, adj_lists in test_loader:
            data = data.to(device)
            output = model(data, adj_lists)
            predictions.append(output.cpu().numpy())
            targets.append(target.squeeze(-1).cpu().numpy())
    
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    
    predictions_orig = y_scaler(predictions)
    targets_orig = y_scaler(targets)
    
    error_percentages = np.abs(predictions_orig - targets_orig) / (np.abs(targets_orig) + 1e-10) * 100
    mean_error_percentage = np.mean(error_percentages)
    
    print("\nTest Set Predictions:")
    print("Sample | Actual Time (ms) | Predicted Time (ms) | Error Percentage (%)")
    print("-" * 60)
    for i, (actual, pred, err) in enumerate(zip(targets_orig, predictions_orig, error_percentages)):
        print(f"{i+1:6d} | {actual:15.4f} | {pred:18.4f} | {err:20.4f}")
    
    return predictions_orig, targets_orig, error_percentages, mean_error_percentage

def plot_loss(train_losses, val_losses, output_file='loss_plot_graph_lstm.png'):
    """
    Plot training and validation loss.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss (Graph-LSTM)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def main():
    # Load dataset
    sequences, execution_times = load_dataset()
    
    # Prepare splits
    train_dataset, val_dataset, test_dataset, y_scaler, seq_len = prepare_train_val_test_split(
        sequences, execution_times
    )
    
    # Create dataloaders with optimizations
    collator = GraphAttentionCollator()
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, 
        collate_fn=collator, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, 
        collate_fn=collator, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, 
        collate_fn=collator, num_workers=4, pin_memory=True
    )
    
    # Initialize model
    input_dim = sequences.shape[2]
    print(f"Main: input_dim={input_dim}, seq_len={seq_len}")
    model = GraphLSTM(input_dim=input_dim, seq_len=seq_len, hidden_dim=256).to(device)
    
    # Train model
    train_losses, val_losses = train_model_with_fold(
        model, train_loader, val_loader,
        use_amp=torch.cuda.is_available()
    )
    
    # Evaluate model
    predictions, targets, error_percentages, mean_error_percentage = evaluate_model(
        model, test_loader, device, y_scaler
    )
    print(f"\nMean Error Percentage: {mean_error_percentage:.4f}%")
    
    # Plot loss
    plot_loss(train_losses, val_losses)
    print("Loss plot saved as 'loss_plot_graph_lstm.png'")

if __name__ == "__main__":
    main()
