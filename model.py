import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

# Custom Dataset class
class ScheduleDataset(Dataset):
    def __init__(self, sequences, execution_times):
        self.sequences = torch.FloatTensor(sequences.astype(np.float32))  # Shape: (samples, timesteps, features)
        self.execution_times = torch.FloatTensor(execution_times).view(-1, 1)  # Shape: (samples, 1)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.execution_times[idx]

# LSTM with Attention model
class LSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super(LSTMAttention, self).__init__()
        self.hidden_dim = hidden_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim, 1)  # Simple attention scoring
        self.dropout = nn.Dropout(dropout)
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch_size, timesteps, hidden_dim)
        lstm_out = self.dropout(lstm_out)
        
        attention_scores = self.attention(lstm_out)  # (batch_size, timesteps, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, timesteps, 1)
        
        context = torch.sum(lstm_out * attention_weights, dim=1)  # (batch_size, hidden_dim)
        context = self.norm(context)
        
        out = self.fc1(context)  # (batch_size, 32)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)  # (batch_size, output_dim)
        return out

# Load the preprocessed dataset
def load_dataset(data_dir="preprocessed_dataset"):
    sequence_data = np.load(f"{data_dir}/sequence_data.npy", allow_pickle=True)
    if sequence_data.dtype == object:
        sequence_data = np.stack(sequence_data).astype(np.float32)
    else:
        sequence_data = sequence_data.astype(np.float32)
    
    edge_df = pd.read_csv(f"{data_dir}/edge_features.csv")
    node_df = pd.read_csv(f"{data_dir}/node_features.csv")
    execution_times = np.load(f"{data_dir}/execution_times.npy", allow_pickle=True).astype(np.float32)
    return sequence_data, edge_df, node_df, execution_times

# Train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * sequences.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * sequences.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if epoch > 10 and val_losses[-1] > val_losses[-2]:
            print("Early stopping triggered.")
            break
    
    return train_losses, val_losses

# Plot and save loss
def plot_and_save_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_new.png")
    plt.close()
    print("Loss plot saved as loss_new.png")

# Main execution
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    sequence_data, edge_df, node_df, execution_times = load_dataset()
    print("Sequence Data Shape:", sequence_data.shape)
    print("Execution Times Shape:", execution_times.shape)
    
    # Normalize execution times
    scaler = StandardScaler()
    execution_times_scaled = scaler.fit_transform(execution_times.reshape(-1, 1)).flatten()
    
    # Split into train+val and holdout test set (10 samples)
    X_temp, X_holdout, y_temp, y_holdout = train_test_split(
        sequence_data, execution_times_scaled, test_size=10, random_state=42
    )
    # Further split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.222, random_state=42  # ~20% of remaining for validation
    )
    
    print("Train Shape:", X_train.shape, y_train.shape)
    print("Validation Shape:", X_val.shape, y_val.shape)
    print("Holdout Test Shape:", X_holdout.shape, y_holdout.shape)
    
    # Create datasets and dataloaders
    train_dataset = ScheduleDataset(X_train, y_train)
    val_dataset = ScheduleDataset(X_val, y_val)
    test_dataset = ScheduleDataset(X_holdout, y_holdout)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)  # Batch size = 10 for holdout
    
    # Model parameters
    input_dim = X_train.shape[2]  # Number of features (11)
    hidden_dim = 64
    output_dim = 1  # Single execution time prediction
    
    # Initialize model, loss, and optimizer
    model = LSTMAttention(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, device=device)
    
    # Evaluate on holdout test set and calculate error percentage
    model.eval()
    with torch.no_grad():
        y_pred_scaled = []
        y_true_scaled = []
        for sequences, targets in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            y_pred_scaled.append(outputs.cpu().numpy())
            y_true_scaled.append(targets.numpy())
        
        y_pred_scaled = np.concatenate(y_pred_scaled).flatten()
        y_true_scaled = np.concatenate(y_true_scaled).flatten()
        
        # Inverse transform to original scale
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
        
        # Calculate MAE and percentage error for each sample
        test_mae = np.mean(np.abs(y_true - y_pred))
        print(f"\nTest MAE (original scale, 10 holdout samples): {test_mae:.4f} ms")
        
        print("\nPredictions for 10 Holdout Samples:")
        print("Sample | True Time (ms) | Predicted Time (ms) | Error (%)")
        print("-" * 60)
        for i in range(len(y_true)):
            true_time = y_true[i]
            pred_time = y_pred[i]
            error_percent = abs(true_time - pred_time) / true_time * 100 if true_time != 0 else 0
            print(f"{i+1:6d} | {true_time:13.4f} | {pred_time:17.4f} | {error_percent:9.2f}")
        
        # Average error percentage
        avg_error_percent = np.mean([abs(true - pred) / true * 100 if true != 0 else 0 for true, pred in zip(y_true, y_pred)])
        print(f"\nAverage Error Percentage: {avg_error_percent:.2f}%")
    
    # Plot and save loss
    plot_and_save_loss(train_losses, val_losses)
    
    # Save the model
    torch.save(model.state_dict(), "lstm_attention_model.pth")
    print("Model saved to lstm_attention_model.pth")
    
    # Save the scaler
    np.save("execution_time_scaler.npy", [scaler.mean_, scaler.scale_])
    print("Scaler saved to execution_time_scaler.npy")
