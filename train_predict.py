import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

# Custom Dataset class for PyTorch
class ExecutionTimeDataset(Dataset):
    def __init__(self, sequences, exec_times):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.exec_times = torch.tensor(exec_times, dtype=torch.float32)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.exec_times[idx]

# Load the precomputed dataset
def load_dataset():
    halide_sequences = np.load('halide_sequences.npy')
    halide_exec_times = np.load('halide_exec_times.npy')
    tiramisu_sequences = np.load('tiramisu_sequences.npy')
    tiramisu_exec_times = np.load('tiramisu_exec_times.npy')
    
    # Combine datasets
    X = np.concatenate([halide_sequences, tiramisu_sequences], axis=0)
    y = np.concatenate([halide_exec_times, tiramisu_exec_times], axis=0)
    
    print("Combined Sequences Shape:", X.shape)  # (n_samples, 50, 44)
    print("Combined Execution Times Shape:", y.shape)  # (n_samples,)
    
    return X, y

# Preprocess data
def preprocess_data(X, y):
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_scaled, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)
    
    print("Train Shape:", X_train.shape, y_train.shape)
    print("Validation Shape:", X_val.shape, y_val.shape)
    print("Test Shape:", X_test.shape, y_test.shape)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

# Define the LSTM model
class ExecutionTimePredictor(nn.Module):
    def __init__(self, input_dim=44, hidden_dim1=128, hidden_dim2=64, dropout=0.3):
        super(ExecutionTimePredictor, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)  # Outputs all timesteps by default
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim2, 32)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout - 0.1)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        out, _ = self.lstm1(x)  # out: (batch_size, seq_len, hidden_dim1)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)  # out: (batch_size, seq_len, hidden_dim2)
        out = self.dropout2(out[:, -1, :])  # Take last timestep: (batch_size, hidden_dim2)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc2(out)
        return out

# Training function
def train_model(model, train_loader, val_loader, device, epochs=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * sequences.size(0)
            train_mae += torch.abs(outputs.squeeze() - targets).sum().item()
        
        train_loss /= len(train_loader.dataset)
        train_mae /= len(train_loader.dataset)
        train_losses.append(train_loss)
        train_maes.append(train_mae)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item() * sequences.size(0)
                val_mae += torch.abs(outputs.squeeze() - targets).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}")
    
    return train_losses, val_losses, train_maes, val_maes

# Plot training history
def plot_history(train_losses, val_losses, train_maes, val_maes):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_maes, label='Train MAE')
    plt.plot(val_maes, label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history_pytorch.png')
    plt.close()

# Main execution
def main():
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and preprocess data
    X, y = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess_data(X, y)
    
    # Create datasets and dataloaders
    train_dataset = ExecutionTimeDataset(X_train, y_train)
    val_dataset = ExecutionTimeDataset(X_val, y_val)
    test_dataset = ExecutionTimeDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = ExecutionTimePredictor().to(device)
    
    # Train model
    train_losses, val_losses, train_maes, val_maes = train_model(model, train_loader, val_loader, device)
    
    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    test_mae = 0.0
    y_pred_scaled = []
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            loss = nn.MSELoss()(outputs.squeeze(), targets)
            test_loss += loss.item() * sequences.size(0)
            test_mae += torch.abs(outputs.squeeze() - targets).sum().item()
            y_pred_scaled.extend(outputs.squeeze().cpu().numpy())
    
    test_loss /= len(test_loader.dataset)
    test_mae /= len(test_loader.dataset)
    print(f"\nTest Loss (MSE): {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Inverse transform predictions
    y_pred_scaled = np.array(y_pred_scaled)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    print("\nSample Predictions (seconds):", y_pred[:5])
    print("Sample Actuals (seconds):", y_test_original[:5])
    
    # Plot training history
    plot_history(train_losses, val_losses, train_maes, val_maes)
    
    # Save model and scaler
    torch.save(model.state_dict(), 'execution_predictor_pytorch.pt')
    np.save('scaler_mean.npy', scaler.mean_)
    np.save('scaler_scale.npy', scaler.scale_)
    print("\nModel and scaler saved.")

if __name__ == "__main__":
    # Ensure data files exist
    required_files = ['halide_sequences.npy', 'halide_exec_times.npy', 
                      'tiramisu_sequences.npy', 'tiramisu_exec_times.npy']
    for file in required_files:
        if not os.path.exists(file):
            print(f"Error: {file} not found. Run transformer.py first to generate the dataset.")
            exit(1)
    
    main()
