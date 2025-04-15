import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants (must match dataset creation)
MAX_NODES = 50
MAX_EDGES = 50
NODE_FEATURE_DIM = 67  # 24 ops + 8 access patterns + 35 scheduling features
EDGE_FEATURE_DIM = 80  # 16 footprint + 64 Jacobian
HIDDEN_DIM = 128
NUM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HalideDataset(Dataset):
    """
    Custom Dataset for loading representation_halide.pkl.
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Normalize inputs
        node_tensor = (item["node_tensor"] - self.node_mean) / self.node_std
        edge_tensor = (item["edge_tensor"] - self.edge_mean) / self.edge_std
        exec_time = (item["execution_time"].item() - self.time_mean) / self.time_std
        # Ensure correct shapes
        logging.debug(f"__getitem__ shapes - Node: {node_tensor.shape}, Edge: {edge_tensor.shape}")
        return (
            node_tensor,  # [MAX_NODES, NODE_FEATURE_DIM]
            edge_tensor,  # [MAX_EDGES, EDGE_FEATURE_DIM]
            torch.tensor([exec_time], dtype=torch.float32)  # [1]
        )

class ExecutionTimeLSTM(nn.Module):
    """
    LSTM model to predict execution time from node and edge tensors.
    """
    def __init__(self, node_input_dim=NODE_FEATURE_DIM, edge_input_dim=EDGE_FEATURE_DIM,
                 hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS):
        super(ExecutionTimeLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Node LSTM
        self.node_lstm = nn.LSTM(node_input_dim, hidden_dim, num_layers, batch_first=True)
        # Edge LSTM
        self.edge_lstm = nn.LSTM(edge_input_dim, hidden_dim, num_layers, batch_first=True)
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        
    def forward(self, node_tensor, edge_tensor):
        # Handle unexpected dimensions
        if node_tensor.dim() == 4 and node_tensor.size(1) == 1:
            node_tensor = node_tensor.squeeze(1)  # [batch_size, 1, MAX_NODES, NODE_FEATURE_DIM] -> [batch_size, MAX_NODES, NODE_FEATURE_DIM]
            edge_tensor = edge_tensor.squeeze(1)  # [batch_size, 1, MAX_EDGES, EDGE_FEATURE_DIM] -> [batch_size, MAX_EDGES, EDGE_FEATURE_DIM]
        elif node_tensor.dim() == 2:
            node_tensor = node_tensor.unsqueeze(0)  # [MAX_NODES, NODE_FEATURE_DIM] -> [1, MAX_NODES, NODE_FEATURE_DIM]
            edge_tensor = edge_tensor.unsqueeze(0)  # [MAX_EDGES, EDGE_FEATURE_DIM] -> [1, MAX_EDGES, EDGE_FEATURE_DIM]
        elif node_tensor.dim() != 3:
            raise ValueError(f"Expected 3D node_tensor after correction, got shape {node_tensor.shape}")
        
        batch_size = node_tensor.size(0)
        
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(node_tensor.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(node_tensor.device)
        
        # Node LSTM
        node_out, _ = self.node_lstm(node_tensor, (h0, c0))
        node_repr = node_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # Edge LSTM
        edge_out, _ = self.edge_lstm(edge_tensor, (h0, c0))
        edge_repr = edge_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # Combine representations
        combined = torch.cat((node_repr, edge_repr), dim=1)
        x = self.relu(self.fc1(combined))
        x = self.fc2(x)
        return x.squeeze(-1)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, model_path="best_model.pth"):
    """
    Train the LSTM model and save the best model based on validation loss.
    """
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for node_tensor, edge_tensor, exec_time in train_loader:
            node_tensor, edge_tensor, exec_time = node_tensor.to(DEVICE), edge_tensor.to(DEVICE), exec_time.to(DEVICE)
            # Log shapes for debugging
            logging.debug(f"Batch shapes - Node: {node_tensor.shape}, Edge: {edge_tensor.shape}, Exec: {exec_time.shape}")
            
            optimizer.zero_grad()
            output = model(node_tensor, edge_tensor)
            loss = criterion(output, exec_time.squeeze(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * node_tensor.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for node_tensor, edge_tensor, exec_time in val_loader:
                node_tensor, edge_tensor, exec_time = node_tensor.to(DEVICE), edge_tensor.to(DEVICE), exec_time.to(DEVICE)
                output = model(node_tensor, edge_tensor)
                loss = criterion(output, exec_time.squeeze(-1))
                val_loss += loss.item() * node_tensor.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            logging.info(f"Saved best model with Val Loss: {val_loss:.6f}")

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
        for node_tensor, edge_tensor, exec_time in test_loader:
            node_tensor, edge_tensor = node_tensor.to(DEVICE), edge_tensor.to(DEVICE)
            output = model(node_tensor, edge_tensor)
            # Denormalize predictions
            output = output * dataset.time_std + dataset.time_mean
            exec_time = exec_time.squeeze(-1) * dataset.time_std + dataset.time_mean
            predictions.extend(output.cpu().numpy())
            actuals.extend(exec_time.cpu().numpy())
            mse += ((output - exec_time) ** 2).sum().item()
            mae += torch.abs(output - exec_time).sum().item()
    
    mse /= len(test_loader.dataset)
    mae /= len(test_loader.dataset)
    logging.info(f"Test MSE: {mse:.6f}, Test MAE: {mae:.6f}")
    return predictions, actuals

def main():
    # Load dataset
    dataset_path = "representation_halide.pkl"
    if not os.path.exists(dataset_path):
        logging.error(f"Dataset {dataset_path} does not exist.")
        return
    
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
    
    logging.info(f"Loaded {len(data)} samples from {dataset_path}")
    
    # Split data
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)  # 60% train, 20% val, 20% test
    
    train_dataset = HalideDataset(train_data)
    val_dataset = HalideDataset(train_data)  # Use train stats for consistency
    test_dataset = HalideDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = ExecutionTimeLSTM().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train
    logging.info("Starting training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)
    
    # Evaluate
    logging.info("Evaluating model...")
    predictions, actuals = evaluate_model(model, test_loader, test_dataset)
    
    # Example predictions
    for i in range(min(5, len(predictions))):
        logging.info(f"Sample {i+1}: Predicted {predictions[i]:.2f} ms, Actual {actuals[i]:.2f} ms")

if __name__ == "__main__":
    main()
