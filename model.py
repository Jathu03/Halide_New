import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import json

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 1. Load the preprocessed data
def load_preprocessed_data(data_dir="preprocessed_dataset"):
    try:
        sequence_data = np.load(os.path.join(data_dir, "sequence_data.npy"), allow_pickle=True)
        execution_times = np.load(os.path.join(data_dir, "execution_times.npy"), allow_pickle=True)
        
        sequence_data = sequence_data.astype(np.float32)
        execution_times = execution_times.astype(np.float32)
        
        if sequence_data.size == 0 or execution_times.size == 0:
            raise ValueError("Loaded data is empty. Check the .npy files.")
        
        return sequence_data, execution_times  # (7999, 38, 11), (7999,)
    except Exception as e:
        print(f"Error loading data from {data_dir}: {e}")
        raise

# 2. Custom Dataset for PyTorch
class ExecutionTimeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)  # y will be (n_samples, 10)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 3. Prepare data with train/val split
def prepare_lstm_data(sequence_data, execution_times, batch_size=32):
    scaler_X = StandardScaler()
    n_samples, seq_len, n_features = sequence_data.shape
    X_reshaped = sequence_data.reshape(-1, n_features)  # (7999*38, 11)
    X_scaled = scaler_X.fit_transform(X_reshaped).reshape(n_samples, seq_len, n_features)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(execution_times.reshape(-1, 1)).flatten()  # (7999,)
    y_scaled_10 = np.tile(y_scaled[:, np.newaxis], (1, 10))  # (7999, 10)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled_10, test_size=0.2, random_state=42
    )

    train_dataset = ExecutionTimeDataset(X_train, y_train)
    val_dataset = ExecutionTimeDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_train, X_val, y_train, y_val, scaler_X, scaler_y

# 4. Define LSTM model for scripting
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=512, hidden_size2=256, hidden_size3=128, dropout=0.3, num_heads=4):
        super(LSTMModel, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.num_heads = num_heads
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True, bidirectional=True)
        self.bn1 = nn.BatchNorm1d(hidden_size1 * 2)
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(hidden_size1 * 2, hidden_size2, batch_first=True, bidirectional=True)
        self.bn2 = nn.BatchNorm1d(hidden_size2 * 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.lstm3 = nn.LSTM(hidden_size2 * 2, hidden_size3, batch_first=True, bidirectional=True)
        self.bn3 = nn.BatchNorm1d(hidden_size3 * 2)
        self.dropout3 = nn.Dropout(dropout)
        
        # Residual projection layers
        self.residual_proj1 = nn.Linear(input_size, hidden_size1 * 2)
        self.residual_proj2 = nn.Linear(hidden_size1 * 2, hidden_size2 * 2)
        self.residual_proj3 = nn.Linear(hidden_size2 * 2, hidden_size3 * 2)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size3 * 2, num_heads=num_heads, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size3 * 2, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.dropout_fc2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(64, 10)  # 10 outputs for 10 schedules
    
    def forward(self, x):
        # x: (batch_size, 38, input_size)
        out1, _ = self.lstm1(x)
        out1 = self.bn1(out1.transpose(1, 2)).transpose(1, 2)
        out1 = self.dropout1(out1)
        residual1 = self.residual_proj1(x)
        out1 = out1 + residual1
        
        out2, _ = self.lstm2(out1)
        out2 = self.bn2(out2.transpose(1, 2)).transpose(1, 2)
        out2 = self.dropout2(out2)
        residual2 = self.residual_proj2(out1)
        out2 = out2 + residual2
        
        out3, _ = self.lstm3(out2)
        out3 = self.bn3(out3.transpose(1, 2)).transpose(1, 2)
        out3 = self.dropout3(out3)
        residual3 = self.residual_proj3(out2)
        out3 = out3 + residual3
        
        # Multi-head self-attention
        attn_output, _ = self.attention(out3, out3, out3)
        out = torch.mean(attn_output, dim=1)
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.bn_fc1(out)
        out = self.relu1(out)
        out = self.dropout_fc1(out)
        
        out = self.fc2(out)
        out = self.bn_fc2(out)
        out = self.relu2(out)
        out = self.dropout_fc2(out)
        
        out = self.fc3(out)
        return out

# 5. Train the model and save with scripting
def train_model(model, train_loader, val_loader, device, epochs=300, patience=20):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_losses = []
    val_losses = []
    
    os.makedirs("loss_model", exist_ok=True)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = "best_lstm_model.pt"
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()  # Save state_dict instead of tracing here
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # After training, load best weights and script the model
    model.load_state_dict(best_model_state)
    model.eval()
    scripted_model = torch.jit.script(model)
    scripted_model.save(best_model_path)
    print(f"Scripted model saved to {best_model_path}")
    
    np.save("loss_model/train_losses.npy", np.array(train_losses))
    np.save("loss_model/val_losses.npy", np.array(val_losses))
    
    return train_losses, val_losses, best_model_path

# 6. Evaluate and predict for 10 schedules
def evaluate_model(model, X_val, y_val, scaler_y, device):
    model.eval()
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred = model(X_val_tensor).cpu().numpy()
    
    n_val = y_val.shape[0]
    y_val_orig = scaler_y.inverse_transform(y_val.reshape(-1, 1)).reshape(n_val, 10)
    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(n_val, 10)
    
    error_percentages = np.abs(y_val_orig - y_pred_orig) / np.maximum(y_val_orig, 1e-6) * 100
    mean_error_percentage = np.mean(error_percentages)
    print(f"Mean Error Percentage across 10 schedules: {mean_error_percentage:.4f}%")
    
    print("\nSample Predictions (first 5 samples, first 3 schedules):")
    for i in range(min(5, n_val)):
        print(f"Sample {i+1}: Actual={y_val_orig[i, 0]:.4f} ms, Predicted={y_pred_orig[i, 0]:.4f} ms, "
              f"Error%={error_percentages[i, 0]:.4f}% (Schedule 1)")
    
    return y_val_orig, y_pred_orig, error_percentages

# 7. Plot train and validation losses
def plot_results(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig("loss_model.png")
    plt.close()
    print("Loss plot saved as loss_model.png")

# 8. Save scalers
def save_scalers(scaler_X, scaler_y):
    scaler_X_data = {
        "means": scaler_X.mean_.tolist(),
        "scales": scaler_X.scale_.tolist(),
        "feature_names": [f"feature_{i}" for i in range(len(scaler_X.mean_))]
    }
    with open("scaler_X.json", "w") as f:
        json.dump(scaler_X_data, f)
    print("Scaler X saved to scaler_X.json")

    scaler_y_data = {
        "mean": float(scaler_y.mean_[0]),
        "scale": float(scaler_y.scale_[0]),
        "is_log_transformed": False
    }
    with open("scaler_y.json", "w") as f:
        json.dump(scaler_y_data, f)
    print("Scaler Y saved to scaler_y.json")

# Main execution
if __name__ == "__main__":
    # Load data
    data_dir = "preprocessed_dataset"
    sequence_data, execution_times = load_preprocessed_data(data_dir)
    
    print("Sequence Data Shape:", sequence_data.shape)
    print("Execution Times Shape:", execution_times.shape)

    if sequence_data.size == 0 or execution_times.size == 0:
        print("Error: No valid data loaded. Please check the preprocessing step.")
        exit()

    # Prepare data
    batch_size = 32
    train_loader, val_loader, X_train, X_val, y_train, y_val, scaler_X, scaler_y = prepare_lstm_data(
        sequence_data, execution_times, batch_size
    )
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = sequence_data.shape[2]  # 11 features
    model = LSTMModel(input_size=input_size).to(device)
    
    # Print model summary
    print(model)
    
    # Train model
    train_losses, val_losses, best_model_path = train_model(
        model, train_loader, val_loader, device, epochs=300, patience=20
    )
    
    # Load best scripted model
    model = torch.jit.load(best_model_path).to(device)
    
    # Evaluate and predict
    y_val_orig, y_pred_orig, error_percentages = evaluate_model(model, X_val, y_val, scaler_y, device)
    
    # Plot results
    plot_results(train_losses, val_losses)
    
    # Save the final model state dict (optional)
    torch.save(model.state_dict(), "lstm_execution_time_model.pth")
    print("Model state dict saved to lstm_execution_time_model.pth")
    model.save("lstm_model.pt")
    print("Scripted model saved to lstm_model.pt")

    # Save scalers
    save_scalers(scaler_X, scaler_y)
