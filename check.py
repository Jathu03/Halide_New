import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recursive function to flatten and encode JSON data with improved encoding
def flatten_json(data, parent_key='', sep='_'):
    items = []
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            items.extend(flatten_json(value, new_key, sep=sep))
    elif isinstance(data, list):
        for i, value in enumerate(data):
            new_key = f"{parent_key}{sep}{i}"
            items.extend(flatten_json(value, new_key, sep=sep))
    else:
        try:
            value = float(data)
        except (ValueError, TypeError):
            # Improved categorical encoding - one-hot for common values
            if isinstance(data, str):
                # Create a more meaningful hash for strings
                value = hash(str(data)) / 1e10  # Normalize hash value
            else:
                value = 0.0  # Default for non-numeric, non-string
        items.append((parent_key, value))
    return items

# Extract target variable (total_execution_time_ms)
def extract_target(data):
    for item in data.get("scheduling_data", []):
        if item.get("name") == "total_execution_time_ms":
            return item["value"]
    return None

# Prepare sequences for LSTM with improved structure
def prepare_sequences(flattened_data):
    sequences = {}
    for key, value in flattened_data:
        top_key = key.split('_')[0]
        if top_key not in sequences:
            sequences[top_key] = []
        sequences[top_key].append(value)
    
    # Convert to fixed-length feature vectors by padding or truncating
    max_seq_len = 50  # Set a reasonable maximum sequence length
    tensor_sequences = []
    
    for seq in sequences.values():
        # Pad or truncate to fixed length
        if len(seq) > max_seq_len:
            seq = seq[:max_seq_len]
        else:
            seq = seq + [0.0] * (max_seq_len - len(seq))
        
        tensor_seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(1)  # [seq_len, 1]
        tensor_sequences.append(tensor_seq)
    
    return tensor_sequences

# Custom Dataset with normalization
class ScheduleDataset(Dataset):
    def __init__(self, sequences_list, targets, target_scaler=None, train=True):
        self.sequences_list = sequences_list  # List of lists of sequences
        
        # Normalize targets
        if train:
            self.target_scaler = StandardScaler()
            normalized_targets = self.target_scaler.fit_transform(np.array(targets).reshape(-1, 1)).flatten()
        else:
            normalized_targets = target_scaler.transform(np.array(targets).reshape(-1, 1)).flatten()
            self.target_scaler = target_scaler
            
        self.targets = torch.tensor(normalized_targets, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.sequences_list)

    def __getitem__(self, idx):
        sequences = self.sequences_list[idx]
        target = self.targets[idx]
        return sequences, target

# Custom collate function for variable-length sequences
def custom_collate_fn(batch):
    sequences_list = [item[0] for item in batch]  # List of sequence lists
    targets = torch.stack([item[1] for item in batch])  # Stack targets
    return sequences_list, targets

# Improved LSTM Model with attention mechanism
class ImprovedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(ImprovedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Main LSTM
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Output layers with dropout for regularization
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        # Initial hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward
        lstm_out, _ = self.lstm(x, (h0, c0))  # lstm_out: [batch_size, seq_len, hidden_size]
        
        # Apply attention
        attention_weights = self.attention(lstm_out).squeeze(-1)  # [batch_size, seq_len]
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention weights
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1), 
            lstm_out
        ).squeeze(1)  # [batch_size, hidden_size]
        
        # Final prediction through fully connected layers
        out = self.dropout(context_vector)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out

    def process_sequences(self, sequences_list):
        batch_size = len(sequences_list)
        
        # Process each sample in the batch
        outputs = []
        for sequences in sequences_list:
            # Concatenate sequences along feature dimension for each sample
            # Each sequence is [seq_len, 1], we want to make it [seq_len, num_sequences]
            sample_tensor = torch.cat(sequences, dim=1).to(device)  # [seq_len, num_sequences]
            sample_tensor = sample_tensor.transpose(0, 1).unsqueeze(0)  # [1, num_sequences, seq_len]
            outputs.append(sample_tensor)
        
        # Combine all samples in batch
        if outputs:
            batch_tensor = torch.cat(outputs, dim=0)  # [batch_size, num_sequences, seq_len]
            return self.forward(batch_tensor)
        else:
            return torch.zeros(batch_size, 1).to(device)

# Load and preprocess all data
def load_synthetic_data(folder_path):
    all_sequences = []
    all_targets = []
    
    for program_folder in os.listdir(folder_path):
        program_path = os.path.join(folder_path, program_folder)
        if not os.path.isdir(program_path):
            continue
        
        for schedule_file in os.listdir(program_path):
            if not schedule_file.endswith('.json'):
                continue
            
            file_path = os.path.join(program_path, schedule_file)
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            flattened_data = flatten_json(data)
            sequences = prepare_sequences(flattened_data)
            target = extract_target(data)
            
            if target is not None:
                all_sequences.append(sequences)
                all_targets.append(target)
    
    return all_sequences, all_targets

# Training function with early stopping and lr scheduling
def train_model(model, train_loader, val_loader, num_epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for sequences_list, targets in train_loader:
            optimizer.zero_grad()
            outputs = model.process_sequences(sequences_list)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences_list, targets in val_loader:
                outputs = model.process_sequences(sequences_list)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                # Load best model
                model.load_state_dict(torch.load('best_model.pth'))
                break

# Prediction function with denormalization
def predict_and_evaluate(model, sequences_list, true_targets, target_scaler, num_samples=10):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for sequences in sequences_list[:num_samples]:
            output = model.process_sequences([sequences])  # Process single sample
            # Denormalize the prediction
            norm_pred = output.cpu().numpy().reshape(-1, 1)
            pred = target_scaler.inverse_transform(norm_pred).item()
            predictions.append(pred)
    
    true_targets = true_targets[:num_samples]
    
    # Calculate various error metrics
    abs_errors = [abs(pred - true) for pred, true in zip(predictions, true_targets)]
    rel_errors = [abs(pred - true) / true * 100 for pred, true in zip(predictions, true_targets)]
    mse = sum((pred - true) ** 2 for pred, true in zip(predictions, true_targets)) / len(predictions)
    mae = sum(abs_errors) / len(abs_errors)
    mape = sum(rel_errors) / len(rel_errors)
    
    return predictions, true_targets, rel_errors, {'MSE': mse, 'MAE': mae, 'MAPE': mape}

# Main execution
def main():
    # Hyperparameters
    input_size = 1  # Feature dimension
    hidden_size = 128  # Increased from 64
    num_layers = 3    # Increased from 2
    output_size = 1   # Predicting total_execution_time_ms
    batch_size = 16   # Increased from 4
    num_epochs = 50
    
    # Load data
    folder_path = 'synthetic_data'
    all_sequences, all_targets = load_synthetic_data(folder_path)
    
    if not all_sequences:
        print("No valid data found.")
        return
    
    # Split into train and validation sets
    train_seqs, val_seqs, train_targets, val_targets = train_test_split(
        all_sequences, all_targets, test_size=0.2, random_state=42
    )
    
    # Create datasets with normalization
    train_dataset = ScheduleDataset(train_seqs, train_targets, train=True)
    target_scaler = train_dataset.target_scaler
    val_dataset = ScheduleDataset(val_seqs, val_targets, target_scaler=target_scaler, train=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    # Initialize improved model
    model = ImprovedLSTM(input_size, hidden_size, num_layers, output_size, dropout=0.3).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params} parameters")
    
    # Train model
    train_model(model, train_loader, val_loader, num_epochs)
    
    # Predict and evaluate
    predictions, true_targets, rel_errors, metrics = predict_and_evaluate(
        model, val_seqs, val_targets, target_scaler
    )
    
    # Print results
    print("\nPredictions and Error Percentages for Sample Schedules:")
    for i, (pred, true, error) in enumerate(zip(predictions, true_targets, rel_errors), 1):
        print(f"Schedule {i}:")
        print(f"  Predicted Time: {pred:.2f} ms")
        print(f"  True Time: {true:.2f} ms")
        print(f"  Error Percentage: {error:.2f}%")
    
    print("\nOverall Metrics:")
    print(f"  Mean Squared Error: {metrics['MSE']:.2f}")
    print(f"  Mean Absolute Error: {metrics['MAE']:.2f} ms")
    print(f"  Mean Absolute Percentage Error: {metrics['MAPE']:.2f}%")

if __name__ == "__main__":
    main()
