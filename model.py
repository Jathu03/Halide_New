import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os

class ExecutionTimeLSTM(nn.Module):
    """
    LSTM model for predicting execution times from graph sequences.
    """
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.3):
        super(ExecutionTimeLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_dim)
        last_out = lstm_out[:, -1, :]  # Take the last time step: (batch_size, hidden_dim)
        output = self.fc(last_out)  # (batch_size, 1)
        return output.squeeze(-1)  # (batch_size,)

def load_dataset(file_path='halide_data.npz'):
    """
    Load the dataset from the .npz file.
    """
    data = np.load(file_path)
    sequences = data['sequences'].astype(np.float32)
    execution_times = data['execution_times'].astype(np.float32)
    return sequences, execution_times

def split_data(sequences, execution_times, test_size=20, val_split=0.2, random_state=42):
    """
    Split data into train, validation, and test sets.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        sequences, execution_times, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_split, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    """
    Create PyTorch DataLoaders for training, validation, and testing.
    """
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, device, epochs=100, patience=10):
    """
    Train the model with early stopping and learning rate scheduling.
    """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = 'best_lstm_model.pth'
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_mae += torch.abs(outputs - y_batch).sum().item()
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    os.remove(best_model_path)  # Clean up
    return train_losses, val_losses

def evaluate_model(model, test_loader, device, scaler):
    """
    Evaluate the model on the test set.
    """
    criterion = nn.MSELoss()
    model.eval()
    test_loss = 0.0
    test_mae = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item() * X_batch.size(0)
            test_mae += torch.abs(outputs - y_batch).sum().item()
            predictions.append(outputs.cpu().numpy())
            targets.append(y_batch.cpu().numpy())
    
    test_loss /= len(test_loader.dataset)
    test_mae /= len(test_loader.dataset)
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    
    # Inverse transform for original scale
    predictions_orig = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    targets_orig = scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
    mae_orig = np.mean(np.abs(predictions_orig - targets_orig))
    
    return test_loss, test_mae, mae_orig

def plot_loss(train_losses, val_losses, output_file='loss_plot_pytorch.png'):
    """
    Plot training and validation loss and save to file.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    sequences, execution_times = load_dataset()
    print(f"Loaded dataset with {sequences.shape[0]} samples")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Execution times shape: {execution_times.shape}")
    
    # Standardize execution times
    time_scaler = StandardScaler()
    execution_times = time_scaler.fit_transform(execution_times.reshape(-1, 1)).flatten()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(sequences, execution_times)
    print(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}, Test samples: {X_test.shape[0]}")
    
    # Create dataloaders
    batch_size = 32
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size
    )
    
    # Initialize model
    input_dim = X_train.shape[2]  # Feature dimension
    model = ExecutionTimeLSTM(input_dim=input_dim).to(device)
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, device)
    
    # Evaluate model
    test_loss, test_mae, mae_orig = evaluate_model(model, test_loader, device, time_scaler)
    print(f"Test Loss (MSE, standardized): {test_loss:.4f}")
    print(f"Test MAE (standardized): {test_mae:.4f}")
    print(f"Test MAE (original scale, ms): {mae_orig:.4f}")
    
    # Plot loss
    plot_loss(train_losses, val_losses)
    print("Loss plot saved as 'loss_plot_pytorch.png'")

if __name__ == "__main__":
    main()
