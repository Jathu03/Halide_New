import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import os
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_mean_pool

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MAX_NODES = 50
MAX_EDGES = 50
NODE_FEATURE_DIM = 70  # Updated
EDGE_FEATURE_DIM = 80
HIDDEN_DIM = 256
NUM_LAYERS = 3
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0005
PATIENCE = 10
DROPOUT = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HalideDataset(Dataset):
    """
    Custom Dataset for loading representation_halide_v2.pkl.
    """
    def __init__(self, data):
        self.data = data
        # Compute normalization stats
        node_tensors = torch.stack([d["node_tensor"] for d in data])
        edge_tensors = torch.stack([d["edge_tensor"] for d in data])
        exec_times = torch.tensor([d["execution_time"].item() for d in data], dtype=torch.float32)
        
        self.node_mean = node_tensors.mean(dim=(0, 1), keepdim=True)
        self.node_std = node_tensors.std(dim=(0, 1), keepdim=True) + 1e-6
        self.edge_mean = edge_tensors.mean(dim=(0, 1), keepdim=True)
        self.edge_std = edge_tensors.std(dim=(0, 1), keepdim=True) + 1e-6
        self.time_mean = exec_times.mean()
        self.time_std = exec_times.std() + 1e-6
        
        # Feature selection: keep features with std > 0.01
        self.node_mask = self.node_std.squeeze() > 0.01
        self.edge_mask = self.edge_std.squeeze() > 0.01
        logging.info(f"Selected {self.node_mask.sum().item()}/{NODE_FEATURE_DIM} node features, "
                     f"{self.edge_mask.sum().item()}/{EDGE_FEATURE_DIM} edge features")
        
        # Log execution time stats
        logging.info(f"Exec time stats - Mean: {self.time_mean:.4f}, Std: {self.time_std:.4f}, "
                     f"Min: {exec_times.min():.4f}, Max: {exec_times.max():.4f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Normalize and select features
        node_tensor = (item["node_tensor"] - self.node_mean) / self.node_std
        edge_tensor = (item["edge_tensor"] - self.edge_mean) / self.edge_std
        node_tensor = node_tensor[:, self.node_mask]
        edge_tensor = edge_tensor[:, self.edge_mask]
        exec_time = (item["execution_time"].item() - self.time_mean) / self.time_std
        
        # Build edge_index from tree
        tree = item["tree"]
        edges = tree.get("edges", [])
        edge_index = []
        for edge in edges:
            src = edge.get("source_id", 0)
            dst = edge.get("target_id", 0)
            if isinstance(src, str) and isinstance(dst, str):
                edge_index.append([int(src.split("_")[-1]), int(dst.split("_")[-1])])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=node_tensor,  # [MAX_NODES, selected_node_features]
            edge_index=edge_index,  # [2, num_edges]
            edge_attr=edge_tensor,  # [MAX_EDGES, selected_edge_features]
            y=torch.tensor([exec_time], dtype=torch.float32)  # [1]
        )
        return data

class ExecutionTimeGNN(nn.Module):
    """
    GNN model to predict execution time.
    """
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super(ExecutionTimeGNN, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        current_dim = node_input_dim
        
        # GCN layers
        for _ in range(num_layers):
            self.convs.append(GCNConv(current_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            current_dim = hidden_dim
        
        # Edge feature processing
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Node feature processing
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.relu(x)
            x = self.dropout(x)
        
        # Global pooling
        node_repr = global_mean_pool(x, batch)
        
        # Edge feature processing
        edge_repr = self.edge_mlp(edge_attr)
        edge_repr = edge_repr.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Combine
        combined = torch.cat((node_repr, edge_repr), dim=1)
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience=PATIENCE, model_path="best_model.pth"):
    """
    Train the GNN model with early stopping.
    """
    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_orig_loss = 0.0
        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y.squeeze(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.num_graphs
            
            # Original scale loss
            output_orig = output * train_loader.dataset.time_std + train_loader.dataset.time_mean
            y_orig = data.y.squeeze(-1) * train_loader.dataset.time_std + train_loader.dataset.time_mean
            orig_loss = criterion(output_orig, y_orig)
            train_orig_loss += orig_loss.item() * data.num_graphs
        
        train_loss /= len(train_loader.dataset)
        train_orig_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_orig_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(DEVICE)
                output = model(data)
                loss = criterion(output, data.y.squeeze(-1))
                val_loss += loss.item() * data.num_graphs
                
                output_orig = output * val_loader.dataset.time_std + val_loader.dataset.time_mean
                y_orig = data.y.squeeze(-1) * val_loader.dataset.time_std + val_loader.dataset.time_mean
                orig_loss = criterion(output_orig, y_orig)
                val_orig_loss += orig_loss.item() * data.num_graphs
        
        val_loss /= len(val_loader.dataset)
        val_orig_loss /= len(val_loader.dataset)
        
        logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Train Orig Loss (ms): {train_orig_loss:.6f}, "
                     f"Val Loss: {val_loss:.6f}, Val Orig Loss (ms): {val_orig_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            logging.info(f"Saved best model with Val Loss: {val_loss:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break

def evaluate_model(model, test_loader, dataset, model_path="best_model.pth"):
    """
    Evaluate the model and compute metrics.
    """
    model.load_state_dict(torch.load(model_path))
    model.eval()
    predictions = []
    actuals = []
    mse = 0.0
    mae = 0.0
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(DEVICE)
            output = model(data)
            output_orig = output * dataset.time_std + dataset.time_mean
            y_orig = data.y.squeeze(-1) * dataset.time_std + dataset.time_mean
            predictions.extend(output_orig.cpu().numpy())
            actuals.extend(y_orig.cpu().numpy())
            mse += ((output_orig - y_orig) ** 2).sum().item()
            mae += torch.abs(output_orig - y_orig).sum().item()
    
    mse /= len(test_loader.dataset)
    mae /= len(test_loader.dataset)
    logging.info(f"Test MSE: {mse:.6f}, Test MAE: {mae:.6f}")
    for i in range(min(5, len(predictions))):
        logging.info(f"Sample {i+1}: Predicted {predictions[i]:.2f} ms, Actual {actuals[i]:.2f} ms")
    return predictions, actuals

def main():
    # Load dataset
    dataset_path = "representation_halide_v2.pkl"
    if not os.path.exists(dataset_path):
        logging.error(f"Dataset {dataset_path} does not exist.")
        return
    
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
    
    logging.info(f"Loaded {len(data)} samples from {dataset_path}")
    
    # Split data
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)
    
    train_dataset = HalideDataset(train_data)
    val_dataset = HalideDataset(val_data)
    test_dataset = HalideDataset(test_data)
    
    if train_dataset.time_std < 1e-4:
        logging.error("Execution time variance too low. Check dataset creation.")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    node_input_dim = train_dataset.node_mask.sum().item()
    edge_input_dim = train_dataset.edge_mask.sum().item()
    model = ExecutionTimeGNN(node_input_dim, edge_input_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train
    logging.info("Starting training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)
    
    # Evaluate
    logging.info("Evaluating model...")
    predictions, actuals = evaluate_model(model, test_loader, test_dataset)

if __name__ == "__main__":
    main()
