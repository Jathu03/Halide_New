import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Function to extract unique feature keys from the first JSON file
def get_feature_keys(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    for item in data:
        if "Details" in item and "scheduling_feature" in item["Details"]:
            return sorted(item["Details"]["scheduling_feature"].keys())
    return []

# Function to extract sequence and target from a JSON file
def extract_features_from_json(json_path, feature_keys):
    with open(json_path, 'r') as f:
        data = json.load(f)
    sequence = []
    target = None
    for item in data:
        if "Details" in item and "scheduling_feature" in item["Details"]:
            features = item["Details"]["scheduling_feature"]
            feature_vector = [float(features.get(key, 0)) for key in feature_keys]
            sequence.append(feature_vector)
        if "name" in item and item["name"] == "total_execution_time_ms":
            target = float(item["value"])
    return sequence, target

# Function to pad sequences
def pad_sequences(sequences, max_len, feature_dim):
    padded_sequences = []
    masks = []
    for seq in sequences:
        seq_len = len(seq)
        padded_seq = seq + [[0] * feature_dim] * (max_len - seq_len)
        mask = [1] * seq_len + [0] * (max_len - seq_len)
        padded_sequences.append(padded_seq)
        masks.append(mask)
    return np.array(padded_sequences), np.array(masks)

# Custom Dataset class
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

# LSTM Model definition
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, mask):
        # x: [batch_size, seq_len, input_dim]
        # mask: [batch_size, seq_len]
        outputs, (hn, cn) = self.lstm(x)  # outputs: [batch_size, seq_len, hidden_dim]
        seq_lengths = mask.sum(dim=1).long() - 1  # Last valid index
        batch_size = x.size(0)
        last_outputs = outputs[torch.arange(batch_size), seq_lengths]
        prediction = self.fc(last_outputs)  # [batch_size, output_dim]
        return prediction

# Main function
def main():
    output_folder = "Output_Programs"
    
    # Get feature keys from the first JSON file
    first_program_folder = os.path.join(output_folder, os.listdir(output_folder)[0])
    first_json_path = os.path.join(first_program_folder, os.listdir(first_program_folder)[0])
    feature_keys = get_feature_keys(first_json_path)
    if not feature_keys:
        raise ValueError("No scheduling_feature found in the first JSON file.")
    
    all_sequences = []
    all_targets = []
    
    # Traverse all program folders and JSON files
    for program_folder in os.listdir(output_folder):
        program_path = os.path.join(output_folder, program_folder)
        if os.path.isdir(program_path):
            for json_file in os.listdir(program_path):
                if json_file.endswith('.json'):
                    json_path = os.path.join(program_path, json_file)
                    sequence, target = extract_features_from_json(json_path, feature_keys)
                    if sequence and target is not None:
                        all_sequences.append(sequence)
                        all_targets.append(target)
    
    if not all_sequences:
        raise ValueError("No valid sequences loaded from the data.")
    
    # Pad sequences
    max_len = max(len(seq) for seq in all_sequences)
    feature_dim = len(feature_keys)
    padded_sequences, masks = pad_sequences(all_sequences, max_len, feature_dim)
    
    # Split into training and testing sets
    train_sequences, test_sequences, train_masks, test_masks, train_targets, test_targets = train_test_split(
        padded_sequences, masks, all_targets, test_size=0.2, random_state=42
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
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}')
    
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
