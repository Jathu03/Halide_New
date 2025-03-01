import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

### Helper Functions

#### Extract Feature Keys
def get_feature_keys(json_path):
    """Extract sorted feature keys from 'scheduling_feature' or 'Op histogram' in a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Handle dictionary structure
    if isinstance(data, dict) and "programming_details" in data:
        nodes = data["programming_details"].get("Nodes", [])
        # Try 'scheduling_feature' first
        for node in nodes:
            if "Details" in node and "scheduling_feature" in node["Details"]:
                return sorted(node["Details"]["scheduling_feature"].keys())
        # Fallback to 'Op histogram'
        for node in nodes:
            if "Details" in node and "Op histogram" in node["Details"]:
                return sorted([line.split(":")[0].strip() for line in node["Details"]["Op histogram"]])
    
    # Handle list structure (for compatibility with your sample)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "programming_details" in item:
                nodes = item["programming_details"].get("Nodes", [])
                for node in nodes:
                    if "Details" in node and "scheduling_feature" in node["Details"]:
                        return sorted(node["Details"]["scheduling_feature"].keys())
                    elif "Details" in node and "Op histogram" in node["Details"]:
                        return sorted([line.split(":")[0].strip() for line in node["Details"]["Op histogram"]])
    
    return []

#### Extract Features and Target
def extract_features_from_json(json_path, feature_keys):
    """
    Extract feature sequences and execution time from a JSON file.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    sequence = []
    execution_time = None
    
    # Handle dictionary structure
    if isinstance(data, dict) and "programming_details" in data:
        nodes = data["programming_details"].get("Nodes", [])
        for node in nodes:
            if "Details" in node:
                if "scheduling_feature" in node["Details"]:
                    features = node["Details"]["scheduling_feature"]
                    feature_vector = [features.get(key, 0) for key in feature_keys]
                    sequence.append(feature_vector)
                elif "Op histogram" in node["Details"]:
                    features = {line.split(":")[0].strip(): float(line.split(":")[1].strip())
                                for line in node["Details"]["Op histogram"]}
                    feature_vector = [features.get(key, 0) for key in feature_keys]
                    sequence.append(feature_vector)
        # Look for execution time in the list items
        if isinstance(data["programming_details"], list):
            for item in data["programming_details"]:
                if isinstance(item, dict) and "name" in item and item["name"] == "total_execution_time_ms":
                    execution_time = item["value"]
        else:
            execution_time = data.get("total_execution_time_ms")
    
    # Handle list structure
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                if "programming_details" in item:
                    nodes = item["programming_details"].get("Nodes", [])
                    for node in nodes:
                        if "Details" in node:
                            if "scheduling_feature" in node["Details"]:
                                features = node["Details"]["scheduling_feature"]
                                feature_vector = [features.get(key, 0) for key in feature_keys]
                                sequence.append(feature_vector)
                            elif "Op histogram" in node["Details"]:
                                features = {line.split(":")[0].strip(): float(line.split(":")[1].strip())
                                            for line in node["Details"]["Op histogram"]}
                                feature_vector = [features.get(key, 0) for key in feature_keys]
                                sequence.append(feature_vector)
                if "name" in item and item["name"] == "total_execution_time_ms":
                    execution_time = item["value"]
    
    if not sequence or execution_time is None:
        return None, None
    return sequence, execution_time

#### Load Data and Calculate Speedup
def load_data(output_folder):
    """Load sequences, execution times, calculate speedup, and return feature keys."""
    all_sequences = []
    all_execution_times = []
    feature_keys = None

    # Step 1: Find feature keys from any JSON file
    for program_folder in os.listdir(output_folder):
        program_path = os.path.join(output_folder, program_folder)
        if os.path.isdir(program_path):
            for json_file in os.listdir(program_path):
                if json_file.endswith('.json'):
                    json_path = os.path.join(program_path, json_file)
                    try:
                        keys = get_feature_keys(json_path)
                        if keys:
                            feature_keys = keys
                            break
                    except Exception as e:
                        print(f"Error processing {json_path}: {e}")
            if feature_keys:
                break
    if not feature_keys:
        # Debug: Print contents of a sample JSON file
        for program_folder in os.listdir(output_folder):
            program_path = os.path.join(output_folder, program_folder)
            if os.path.isdir(program_path):
                for json_file in os.listdir(program_path):
                    if json_file.endswith('.json'):
                        json_path = os.path.join(program_path, json_file)
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                        print(f"Sample JSON content from {json_path}: {json.dumps(data, indent=2)[:1000]}...")
                        break
                break
        raise ValueError("No valid feature keys ('scheduling_feature' or 'Op histogram') found in any JSON file.")

    # Step 2: Load data from all JSON files
    for program_folder in os.listdir(output_folder):
        program_path = os.path.join(output_folder, program_folder)
        if os.path.isdir(program_path):
            program_times = []
            program_sequences = []
            for json_file in os.listdir(program_path):
                if json_file.endswith('.json'):
                    json_path = os.path.join(program_path, json_file)
                    try:
                        sequence, execution_time = extract_features_from_json(json_path, feature_keys)
                        if sequence and execution_time is not None:
                            program_sequences.append(sequence)
                            program_times.append(execution_time)
                    except Exception as e:
                        print(f"Error processing {json_path}: {e}")
            # Calculate speedup within each program
            if program_times:
                baseline_time = max(program_times)  # Max time as baseline
                all_sequences.extend(program_sequences)
                all_execution_times.extend([baseline_time / time for time in program_times])

    if not all_sequences:
        raise ValueError("No valid sequences loaded from the data.")
    return all_sequences, all_execution_times, feature_keys

#### Pad Sequences
def pad_sequences(sequences, max_len, feature_dim):
    """Pad sequences to a uniform length with zeros and create masks."""
    padded_sequences = []
    masks = []
    for seq in sequences:
        seq_len = len(seq)
        padded_seq = seq + [[0] * feature_dim] * (max_len - seq_len)
        mask = [1] * seq_len + [0] * (max_len - seq_len)
        padded_sequences.append(padded_seq)
        masks.append(mask)
    return np.array(padded_sequences), np.array(masks)

### Custom Dataset Class
class ScheduleDataset(Dataset):
    def __init__(self, sequences, masks, targets):
        self.sequences = sequences
        self.masks = masks
        self.targets = targets
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'sequence': torch.tensor(self.sequences[idx], dtype=torch.float32),
            'mask': torch.tensor(self.masks[idx], dtype=torch.float32),
            'target': torch.tensor(self.targets[idx], dtype=torch.float32)
        }

### LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, mask):
        outputs, (hn, cn) = self.lstm(x)
        seq_lengths = mask.sum(dim=1).long() - 1
        batch_size = x.size(0)
        last_outputs = outputs[torch.arange(batch_size), seq_lengths]
        prediction = self.fc(last_outputs)
        return prediction

### Main Execution
def main():
    # Set the output folder path (update this to your actual path)
    output_folder = "Output_Programs"  # Example: "/home/kowrisaan/jathu/Halide_New/Output_Programs"
    
    # Load data
    all_sequences, all_speedups, feature_keys = load_data(output_folder)
    print(f"Loaded {len(all_sequences)} sequences with {len(feature_keys)} features each.")

    # Prepare data: Pad sequences
    max_len = max(len(seq) for seq in all_sequences)
    feature_dim = len(feature_keys)
    padded_sequences, masks = pad_sequences(all_sequences, max_len, feature_dim)
    
    # Split into training and testing sets
    train_sequences, test_sequences, train_masks, test_masks, train_targets, test_targets = train_test_split(
        padded_sequences, masks, all_speedups, test_size=0.2, random_state=42
    )
    
    # Create datasets and data loaders
    train_dataset = ScheduleDataset(train_sequences, train_masks, train_targets)
    test_dataset = ScheduleDataset(test_sequences, test_masks, test_targets)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Define the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_dim=feature_dim, hidden_dim=128, num_layers=2, output_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            sequences = batch['sequence'].to(device)
            masks = batch['mask'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            predictions = model(sequences, masks)
            loss = criterion(predictions.squeeze(), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
    
    # Testing
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            sequences = batch['sequence'].to(device)
            masks = batch['mask'].to(device)
            targets = batch['target'].to(device)
            
            predictions = model(sequences, masks)
            loss = criterion(predictions.squeeze(), targets)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f}')

if __name__ == "__main__":
    main()
