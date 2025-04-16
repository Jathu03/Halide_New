import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os

class Attention(nn.Module):
    """
    Attention mechanism to weigh LSTM outputs.
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch_size, seq_len, hidden_dim)
        attn_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)  # (batch_size, seq_len)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch_size, hidden_dim)
        return context, attn_weights

class EnhancedExecutionTimeLSTM(nn.Module):
    """
    Enhanced LSTM model with bidirectional LSTM, attention, and normalization.
    """
    def __init__(self, input_dim, hidden_dim=512, num_layers=4, dropout=0.4):
        super(EnhancedExecutionTimeLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.ln = nn.LayerNorm(hidden_dim * 2)  # For bidirectional output
        self.attention = Attention(hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim * 2)
        lstm_out = self.ln(lstm_out)
        context, _ = self.attention(lstm_out)  # (batch_size, hidden_dim * 2)
        output = self.fc(context)  # (batch_size, 1)
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

def train_model(model, train_loader, val_loader, device, epochs=200, patience=15):
    """
    Train the model with early stopping and cosine annealing scheduler.
    """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
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
        scheduler.step()
        
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
    os.remove(best_model_path)
    return train_losses, val_losses

def evaluate_model(model, test_loader, device, scaler):
    """
    Evaluate the model on the test set and compute error percentages.
    """
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            predictions.append(outputs.cpu().numpy())
            targets.append(y_batch.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    
    # Inverse transform to original scale
    predictions_orig = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    targets_orig = scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
    
    # Calculate error percentages
    error_percentages = np.abs(predictions_orig - targets_orig) / np.abs(targets_orig) * 100
    mean_error_percentage = np.mean(error_percentages)
    
    # Print predictions and errors
    print("\nTest Set Predictions:")
    print("Sample | Actual Time (ms) | Predicted Time (ms) | Error Percentage (%)")
    print("-" * 60)
    for i, (actual, pred, err) in enumerate(zip(targets_orig, predictions_orig, error_percentages)):
        print(f"{i+1:6d} | {actual:15.4f} | {pred:18.4f} | {err:20.4f}")
    
    return predictions_orig, targets_orig, error_percentages, mean_error_percentage

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
    input_dim = X_train.shape[2]
    model = EnhancedExecutionTimeLSTM(input_dim=input_dim).to(device)
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, device)
    
    # Evaluate model
    predictions, targets, error_percentages, mean_error_percentage = evaluate_model(
        model, test_loader, device, time_scaler
    )
    print(f"\nMean Error Percentage: {mean_error_percentage:.4f}%")
    
    # Plot loss
    plot_loss(train_losses, val_losses)
    print("Loss plot saved as 'loss_plot_pytorch.png'")

if __name__ == "__main__":
    main()
