import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import OrderedDict
from sklearn.model_selection import train_test_split

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recursive function to flatten and encode JSON data
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
            value = hash(str(data)) % 1000  # Simple categorical encoding
        items.append((parent_key, value))
    return items

# Extract target variable (total_execution_time_ms)
def extract_target(data):
    for item in data.get("scheduling_data", []):
        if item.get("name") == "total_execution_time_ms":
            return item["value"]
    return None

# Prepare sequences for LSTM
def prepare_sequences(flattened_data):
    sequences = {}
    for key, value in flattened_data:
        top_key = key.split('_')[0]
        if top_key not in sequences:
            sequences[top_key] = []
        sequences[top_key].append(value)
    
    tensor_sequences = []
    for seq in sequences.values():
        tensor_seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(1)  # [seq_len, 1]
        tensor_sequences.append(tensor_seq)
    return tensor_sequences

# Custom Dataset
class ScheduleDataset(Dataset):
    def __init__(self, sequences_list, targets):
        self.sequences_list = sequences_list  # List of lists of sequences
        self.targets = torch.tensor(targets, dtype=torch.float32).to(device)

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

# Define Recursive LSTM Model (Simplified without Embedding)
class RecursiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RecursiveLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # Input size matches data
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden=None):
        if hidden is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            hidden = (h0, c0)
        
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # Take the last output
        return out, hidden

    def process_recursive(self, sequences_list):
        batch_outputs = []
        for sequences in sequences_list:  # Iterate over batch
            outputs = []
            hidden = None
            for seq in sequences:  # Iterate over sequences in one sample
                seq = seq.unsqueeze(0).to(device)  # [1, seq_len, input_size]
                out, hidden = self.forward(seq, hidden)
                outputs.append(out)
            # Average outputs for this sample
            sample_output = torch.mean(torch.stack(outputs), dim=0)
            batch_outputs.append(sample_output)
        return torch.stack(batch_outputs), hidden  # Stack batch outputs

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

# Training function
def train_model(model, train_loader, val_loader, num_epochs=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for sequences_list, targets in train_loader:
            optimizer.zero_grad()
            output, _ = model.process_recursive(sequences_list)
            loss = criterion(output.squeeze(), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences_list, targets in val_loader:
                output, _ = model.process_recursive(sequences_list)
                loss = criterion(output.squeeze(), targets)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

# Prediction function for 10 schedules
def predict_and_evaluate(model, sequences_list, true_targets, num_samples=10):
    model.eval()
    predictions = []
    with torch.no_grad():
        for sequences in sequences_list[:num_samples]:
            output, _ = model.process_recursive([sequences])  # Wrap in list for batch-like processing
            predictions.append(output.item())
    
    true_targets = true_targets[:num_samples]
    errors = [abs(pred - true) / true * 100 for pred, true in zip(predictions, true_targets)]
    
    return predictions, true_targets, errors

# Main execution
def main():
    # Hyperparameters
    input_size = 1  # Matches the actual input data size
    hidden_size = 64
    num_layers = 2
    output_size = 1  # Predicting total_execution_time_ms
    batch_size = 4
    num_epochs = 10
    
    # Load data
    folder_path = 'synthetic_data'
    all_sequences, all_targets = load_synthetic_data(folder_path)
    
    if not all_sequences:
        print("No valid data found.")
        return
    
    # Split into train and test
    train_seqs, test_seqs, train_targets, test_targets = train_test_split(
        all_sequences, all_targets, test_size=0.2, random_state=42
    )
    
    # Create datasets and loaders with custom collate function
    train_dataset = ScheduleDataset(train_seqs, train_targets)
    test_dataset = ScheduleDataset(test_seqs, test_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    # Initialize model
    model = RecursiveLSTM(input_size, hidden_size, num_layers, output_size).to(device)
    
    # Train model
    train_model(model, train_loader, test_loader, num_epochs)
    
    # Predict and evaluate
    predictions, true_targets, errors = predict_and_evaluate(model, test_seqs, test_targets)
    
    # Print results
    print("\nPredictions and Error Percentages for 10 Schedules:")
    for i, (pred, true, error) in enumerate(zip(predictions, true_targets, errors), 1):
        print(f"Schedule {i}:")
        print(f"  Predicted Time: {pred:.2f} ms")
        print(f"  True Time: {true:.2f} ms")
        print(f"  Error Percentage: {error:.2f}%")

if __name__ == "__main__":
    main()
