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

# 3. Prepare data with train/val/test split, selecting low execution time test samples
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

    # Filter out zero or negative execution times and sort by execution time
    valid_indices = execution_times > 0
    X_valid = X_scaled[valid_indices]
    y_valid = y_scaled_10[valid_indices]
    exec_times_valid = execution_times[valid_indices]

    # Select test samples from the lower 25th percentile
    percentile_25 = np.percentile(exec_times_valid, 25)
    low_time_indices = np.where(exec_times_valid <= percentile_25)[0]
    test_indices = np.random.choice(low_time_indices, size=test_size, replace=False)
    
    # Create test set
    X_test = X_valid[test_indices]
    y_test = y_valid[test_indices]
    
    # Remaining data for train/val split
    remain_indices = np.setdiff1d(np.arange(len(X_valid)), test_indices)
    X_remain = X_valid[remain_indices]
    y_remain = y_valid[remain_indices]

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

# 4. Define corrected Recursive LSTM model with iterative reduction
class RecursiveLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, lstm_hidden_size=64, dropout=0.2, num_heads=4):
        super(RecursiveLSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_heads = num_heads
        
        # Initial recursive projection from input_size to hidden_size
        self.initial_recursive_fc = nn.Linear(input_size * 2, hidden_size)
        # Recursive layer for subsequent steps
        self.recursive_fc = nn.Linear(hidden_size * 2, hidden_size)
        self.recursive_ln = nn.LayerNorm(hidden_size)
        self.recursive_dropout = nn.Dropout(dropout)
        
        # LSTM layers for sequential properties
        self.lstm1 = nn.LSTM(hidden_size, lstm_hidden_size, batch_first=True, bidirectional=True)
        self.lstm_ln1 = nn.LayerNorm(lstm_hidden_size * 2)
        self.lstm_dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(lstm_hidden_size * 2, lstm_hidden_size, batch_first=True, bidirectional=True)
        self.lstm_ln2 = nn.LayerNorm(lstm_hidden_size * 2)
        self.lstm_dropout2 = nn.Dropout(dropout)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=lstm_hidden_size * 2, num_heads=num_heads, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_hidden_size * 2, 64)
        self.fc_ln1 = nn.LayerNorm(64)
        self.relu1 = nn.ReLU()
        self.fc_dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(64, 32)
        self.fc_ln2 = nn.LayerNorm(32)
        self.relu2 = nn.ReLU()
        self.fc_dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(32, 10)
    
    def iterative_recursive_step(self, x):
        # x: (batch_size, seq_len, feature_size)
        batch_size, seq_len, feature_size = x.size()
        current_seq = x
        
        # First step with initial input size
        while seq_len > 1:
            if seq_len % 2 != 0:
                padding = torch.zeros(batch_size, 1, current_seq.size(-1), device=x.device)
                current_seq = torch.cat([current_seq, padding], dim=1)
                seq_len += 1
            
            # Pairwise combination
            current_seq = current_seq.view(batch_size, seq_len // 2, 2, current_seq.size(-1))
            current_seq = current_seq.reshape(batch_size, seq_len // 2, current_seq.size(-1) * 2)
            
            # Apply appropriate transformation
            if current_seq.size(-1) == self.input_size * 2:  # Initial step
                current_seq = self.initial_recursive_fc(current_seq)
            else:  # Subsequent steps
                current_seq = self.recursive_fc(current_seq)
            
            current_seq = self.recursive_ln(current_seq)
            current_seq = self.relu1(current_seq)
            current_seq = self.recursive_dropout(current_seq)
            
            # Update seq_len for next iteration
            seq_len = current_seq.size(1)
        
        return current_seq
    
    def forward(self, x):
        # Iterative recursive processing
        recursive_out = self.iterative_recursive_step(x)  # (batch_size, reduced_seq_len, hidden_size)
        
        # Sequential LSTM processing
        out, _ = self.lstm1(recursive_out)
        out = self.lstm_ln1(out)
        out = self.lstm_dropout1(out)
        
        out, _ = self.lstm2(out)
        out = self.lstm_ln2(out)
        out = self.lstm_dropout2(out)
        
        # Attention mechanism
        attn_output, _ = self.attention(out, out, out)
        out = torch.mean(attn_output, dim=1)  # (batch_size, lstm_hidden_size * 2)
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.fc_ln1(out)
        out = self.relu1(out)
        out = self.fc_dropout1(out)
        
        out = self.fc2(out)
        out = self.fc_ln2(out)
        out = self.relu2(out)
        out = self.fc_dropout2(out)
        
        out = self.fc3(out)
        return out

# 5. Train the model with gradient accumulation
def train_model(model, train_loader, val_loader, device, epochs=300, patience=20, accum_steps=4):
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_losses = []
    val_losses = []
    
    os.makedirs("loss_model", exist_ok=True)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = "best_lstm_model.pt"
    best_model_state = None
    
    warmup_epochs = 10
    for epoch in range(epochs):
        if epoch < warmup_epochs:
            lr = 1e-6 + (0.0005 - 1e-6) * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler.step(val_loss if epoch > 0 else best_val_loss)

        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for i, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            mse_loss = criterion_mse(outputs, y_batch)
            l1_loss = criterion_l1(outputs, y_batch)
            loss = (mse_loss + 0.1 * l1_loss) / accum_steps
            loss.backward()
            train_loss += loss.item() * X_batch.size(0) * accum_steps
            
            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        
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
    model = RecursiveLSTMModel(input_size=input_size).to(device)
    
    # Train model with gradient accumulation
    train_losses, val_losses, best_model_path = train_model(
        model, train_loader, val_loader, device, epochs=300, patience=20, accum_steps=4
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
