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
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 3. Prepare data with train/val/test split
def prepare_lstm_data(sequence_data, execution_times, batch_size=32):
    scaler_X = StandardScaler()
    n_samples, seq_len, n_features = sequence_data.shape
    X_reshaped = sequence_data.reshape(-1, n_features)  # (7999*38, 11)
    X_scaled = scaler_X.fit_transform(X_reshaped).reshape(n_samples, seq_len, n_features)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(execution_times.reshape(-1, 1)).flatten()  # (7999,)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42
    )

    train_dataset = ExecutionTimeDataset(X_train, y_train)
    val_dataset = ExecutionTimeDataset(X_val, y_val)
    test_dataset = ExecutionTimeDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y

# 4. Define improved LSTM model with attention
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=256, hidden_size2=128, hidden_size3=64, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size2 = hidden_size2
        
        # Bidirectional LSTMs
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True, bidirectional=True)
        self.bn1 = nn.BatchNorm1d(hidden_size1 * 2)  # *2 for bidirectional
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(hidden_size1 * 2, hidden_size2, batch_first=True, bidirectional=True)
        self.bn2 = nn.BatchNorm1d(hidden_size2 * 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.lstm3 = nn.LSTM(hidden_size2 * 2, hidden_size3, batch_first=True, bidirectional=True)
        self.bn3 = nn.BatchNorm1d(hidden_size3 * 2)
        self.dropout3 = nn.Dropout(dropout)
        
        # Attention layer
        self.attention = nn.Linear(hidden_size3 * 2, 1)
        self.fc1 = nn.Linear(hidden_size3 * 2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        # x: (batch_size, 38, input_size)
        out, _ = self.lstm1(x)  # (batch_size, 38, hidden_size1 * 2)
        out = self.bn1(out.transpose(1, 2)).transpose(1, 2)  # Batch norm across features
        out = self.dropout1(out)
        
        out, _ = self.lstm2(out)  # (batch_size, 38, hidden_size2 * 2)
        out = self.bn2(out.transpose(1, 2)).transpose(1, 2)
        out = self.dropout2(out)
        
        out, _ = self.lstm3(out)  # (batch_size, 38, hidden_size3 * 2)
        out = self.bn3(out.transpose(1, 2)).transpose(1, 2)
        out = self.dropout3(out)
        
        # Attention mechanism
        attn_weights = torch.softmax(self.attention(out), dim=1)  # (batch_size, 38, 1)
        out = torch.sum(out * attn_weights, dim=1)  # (batch_size, hidden_size3 * 2)
        
        out = self.fc1(out)  # (batch_size, 64)
        out = self.relu(out)
        out = self.fc2(out)  # (batch_size, 1)
        return out

# 5. Train the model with enhancements
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
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()  # (batch_size,)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                val_loss += criterion(outputs, y_batch).item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.jit.save(torch.jit.trace(model, torch.randn(1, 38, 11).to(device)), best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    np.save("loss_model/train_losses.npy", np.array(train_losses))
    np.save("loss_model/val_losses.npy", np.array(val_losses))
    
    return train_losses, val_losses, best_model_path

# 6. Evaluate and predict
def evaluate_model(model, X_test, y_test, scaler_y, device):
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred = model(X_test_tensor).squeeze().cpu().numpy()  # (n_test,)
    
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    error_percentages = np.abs(y_test_orig - y_pred_orig) / np.maximum(y_test_orig, 1e-6) * 100
    mean_error_percentage = np.mean(error_percentages)
    mae_orig = np.mean(np.abs(y_test_orig - y_pred_orig))
    print(f"Mean Error Percentage: {mean_error_percentage:.4f}%")
    print(f"MAE in original scale (ms): {mae_orig:.4f}")
    
    return y_test_orig, y_pred_orig, error_percentages

# 7. Plot train and validation losses
def plot_results(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')  # Log scale for better visualization
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

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
    train_loader, val_loader, test_loader, X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y = prepare_lstm_data(
        sequence_data, execution_times, batch_size
    )
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
    
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
    
    # Load best model for evaluation
    model = torch.jit.load(best_model_path).to(device)
    
    # Evaluate and predict
    y_test_orig, y_pred_orig, error_percentages = evaluate_model(model, X_test, y_test, scaler_y, device)
    
    # Plot results
    plot_results(train_losses, val_losses)
    
    # Save the final model
    torch.save(model.state_dict(), "lstm_execution_time_model.pth")
    print("Model state dict saved to lstm_execution_time_model.pth")
    model.save("lstm_model.pt")
    print("TorchScript model saved to lstm_model.pt")

    # Save scalers
    save_scalers(scaler_X, scaler_y)
