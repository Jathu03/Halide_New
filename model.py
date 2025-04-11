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

# 3. Prepare data with train/val/test split, selecting smallest execution times for test
def prepare_lstm_data(sequence_data, execution_times, batch_size=32, test_size=10):
    scaler_X = StandardScaler()
    n_samples, seq_len, n_features = sequence_data.shape
    X_reshaped = sequence_data.reshape(-1, n_features)
    X_scaled = scaler_X.fit_transform(X_reshaped).reshape(n_samples, seq_len, n_features)

    # Log transform execution times
    execution_times_log = np.log1p(execution_times)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(execution_times_log.reshape(-1, 1)).flatten()
    y_scaled_10 = np.tile(y_scaled[:, np.newaxis], (1, 10))  # (n_samples, 10)

    # Sort by execution time and select 10 smallest for test set
    sorted_indices = np.argsort(execution_times)
    test_indices = sorted_indices[:test_size]
    remain_indices = sorted_indices[test_size:]

    X_test = X_scaled[test_indices]
    y_test = y_scaled_10[test_indices]
    X_remain = X_scaled[remain_indices]
    y_remain = y_scaled_10[remain_indices]

    # Split remaining data into train and validation (80-20 split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_remain, y_remain, test_size=0.2, random_state=42
    )

    train_dataset = ExecutionTimeDataset(X_train, y_train)
    val_dataset = ExecutionTimeDataset(X_val, y_val)
    test_dataset = ExecutionTimeDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y

# 4. Define improved LSTM model with 4 layers
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=256, hidden_size2=128, hidden_size3=64, hidden_size4=32, dropout=0.2, num_heads=4):
        super(LSTMModel, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.hidden_size4 = hidden_size4
        self.num_heads = num_heads
        
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True, bidirectional=True)
        self.ln1 = nn.LayerNorm(hidden_size1 * 2)
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(hidden_size1 * 2, hidden_size2, batch_first=True, bidirectional=True)
        self.ln2 = nn.LayerNorm(hidden_size2 * 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.lstm3 = nn.LSTM(hidden_size2 * 2, hidden_size3, batch_first=True, bidirectional=True)
        self.ln3 = nn.LayerNorm(hidden_size3 * 2)
        self.dropout3 = nn.Dropout(dropout)
        
        self.lstm4 = nn.LSTM(hidden_size3 * 2, hidden_size4, batch_first=True, bidirectional=True)
        self.ln4 = nn.LayerNorm(hidden_size4 * 2)
        self.dropout4 = nn.Dropout(dropout)
        
        self.residual_proj1 = nn.Linear(input_size, hidden_size1 * 2)
        self.residual_proj2 = nn.Linear(hidden_size1 * 2, hidden_size2 * 2)
        self.residual_proj3 = nn.Linear(hidden_size2 * 2, hidden_size3 * 2)
        self.residual_proj4 = nn.Linear(hidden_size3 * 2, hidden_size4 * 2)
        
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size4 * 2, num_heads=num_heads, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_size4 * 2, 64)
        self.ln_fc1 = nn.LayerNorm(64)
        self.relu1 = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(64, 32)
        self.ln_fc2 = nn.LayerNorm(32)
        self.relu2 = nn.ReLU()
        self.dropout_fc2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(32, 10)
    
    def forward(self, x):
        out1, _ = self.lstm1(x)
        out1 = self.ln1(out1)
        out1 = self.dropout1(out1)
        residual1 = self.residual_proj1(x)
        out1 = out1 + 0.3 * residual1
        
        out2, _ = self.lstm2(out1)
        out2 = self.ln2(out2)
        out2 = self.dropout2(out2)
        residual2 = self.residual_proj2(out1)
        out2 = out2 + 0.3 * residual2
        
        out3, _ = self.lstm3(out2)
        out3 = self.ln3(out3)
        out3 = self.dropout3(out3)
        residual3 = self.residual_proj3(out2)
        out3 = out3 + 0.3 * residual3
        
        out4, _ = self.lstm4(out3)
        out4 = self.ln4(out4)
        out4 = self.dropout4(out4)
        residual4 = self.residual_proj4(out3)
        out4 = out4 + 0.3 * residual4
        
        attn_output, _ = self.attention(out4, out4, out4)
        out = torch.mean(attn_output, dim=1)
        
        out = self.fc1(out)
        out = self.ln_fc1(out)
        out = self.relu1(out)
        out = self.dropout_fc1(out)
        
        out = self.fc2(out)
        out = self.ln_fc2(out)
        out = self.relu2(out)
        out = self.dropout_fc2(out)
        
        out = self.fc3(out)
        return out

# 5. Train the model with cosine annealing
def train_model(model, train_loader, val_loader, device, epochs=300, patience=20):
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
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
            mse_loss = criterion_mse(outputs, y_batch)
            l1_loss = criterion_l1(outputs, y_batch)
            loss = mse_loss + 0.1 * l1_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Lowered max norm
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
                mse_loss = criterion_mse(outputs, y_batch)
                l1_loss = criterion_l1(outputs, y_batch)
                loss = mse_loss + 0.1 * l1_loss
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        scheduler.step()  # Update learning rate with cosine annealing
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    model.load_state_dict(best_model_state)
    model.eval()
    scripted_model = torch.jit.script(model)
    scripted_model.save(best_model_path)
    
    np.save("loss_model/train_losses.npy", np.array(train_losses))
    np.save("loss_model/val_losses.npy", np.array(val_losses))
    
    return train_losses, val_losses, best_model_path

# 6. Evaluate model and calculate error percentage for test set
def evaluate_model(model, test_loader, scaler_y, device):
    model.eval()
    y_pred_list = []
    y_true_list = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch).cpu().numpy()
            y_pred_list.append(y_pred)
            y_true_list.append(y_batch.numpy())
    
    y_pred = np.concatenate(y_pred_list, axis=0)  # (10, 10)
    y_true = np.concatenate(y_true_list, axis=0)  # (10, 10)
    
    # Inverse transform
    y_true_scaled = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_scaled = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_true_orig = np.expm1(y_true_scaled).reshape(10, 10)
    y_pred_orig = np.expm1(y_pred_scaled).reshape(10, 10)
    
    # Calculate error percentages
    error_percentages = np.abs(y_true_orig - y_pred_orig) / np.maximum(y_true_orig, 1e-6) * 100
    mean_error_percentage = np.mean(error_percentages)
    
    print(f"\nMean Error Percentage across 10 test samples and 10 schedules: {mean_error_percentage:.4f}%")
    
    print("\nTest Set Predictions (10 samples, first 3 schedules):")
    for i in range(10):
        print(f"Sample {i+1}:")
        for j in range(min(3, 10)):
            print(f"  Schedule {j+1}: Actual={y_true_orig[i, j]:.4f} ms, "
                  f"Predicted={y_pred_orig[i, j]:.4f} ms, "
                  f"Error%={error_percentages[i, j]:.4f}%")
    
    return y_true_orig, y_pred_orig, error_percentages

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
    plt.savefig("loss_model/loss_plot.png")
    plt.close()
    print("Loss plot saved as loss_model/loss_plot.png")

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
        "is_log_transformed": True
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
        sequence_data, execution_times, batch_size, test_size=10
    )
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = sequence_data.shape[2]  # 11 features
    model = LSTMModel(input_size=input_size).to(device)
    
    # Train model
    train_losses, val_losses, best_model_path = train_model(
        model, train_loader, val_loader, device, epochs=300, patience=20
    )
    
    # Load best scripted model
    model = torch.jit.load(best_model_path).to(device)
    
    # Evaluate on test set
    y_true_orig, y_pred_orig, error_percentages = evaluate_model(model, test_loader, scaler_y, device)
    
    # Plot results
    plot_results(train_losses, val_losses)
    
    # Save the final model
    torch.save(model.state_dict(), "lstm_execution_time_model.pth")
    print("Model state dict saved to lstm_execution_time_model.pth")
    model.save("lstm_model.pt")
    print("Scripted model saved to lstm_model.pt")

    # Save scalers
    save_scalers(scaler_X, scaler_y)
