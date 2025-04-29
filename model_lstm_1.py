import os
import json
import torch
import numpy as np
from torch import nn
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset, DataLoader

class ExecutionTimePredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=3, bidirectional=True)
        self.attention = nn.MultiheadAttention(2*hidden_size, 4)
        self.fc = nn.Sequential(
            nn.Linear(2*hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return self.fc(attn_out[:, -1])

class HalideDataset(Dataset):
    def __init__(self, root_dir, max_samples=30):
        self.features = []
        self.labels = []
        self._process_directory(root_dir, max_samples)
        
    def _process_directory(self, root_dir, max_samples):
        valid_count = 0
        for dirpath, _, filenames in os.walk(root_dir):
            if 'tree_representation.json' in filenames:
                file_path = os.path.join(dirpath, 'tree_representation.json')
                try:
                    with open(file_path) as f:
                        data = json.load(f)
                    
                    if self._validate_data(data):
                        features = self._extract_features(data)
                        self.features.append(features)
                        self.labels.append(data['execution_time_ms'])
                        valid_count += 1
                        
                        if valid_count >= max_samples:
                            break
                except Exception as e:
                    print(f"Skipping {file_path}: {str(e)}")

    def _validate_data(self, data):
        return data.get('execution_time_ms', -1) > 0 and \
               all(k in data for k in ['scheduling_params', 'program_features'])

    def _extract_features(self, data):
        program_feats = [
            data['program_features']['op_histogram']['add'],
            data['program_features']['memory_access_pattern']['stride'],
            data['program_features']['loop_nest_depth']
        ]
        
        sched_feats = [
            data['scheduling_params']['vector_size'],
            data['scheduling_params']['parallel_degree'],
            data['scheduling_params']['tile_size']
        ]
        
        return np.concatenate([program_feats, sched_feats])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor([self.labels[idx]])

def train_model():
    # Initialize dataset and split
    full_dataset = HalideDataset("Tree_Output")
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # Model configuration
    model = ExecutionTimePredictor(input_size=6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.HuberLoss()
    
    # Training loop
    for epoch in range(100):
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features.unsqueeze(1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        errors = []
        with torch.no_grad():
            for features, labels in test_loader:
                outputs = model(features.unsqueeze(1))
                errors.extend((outputs.squeeze() - labels.squeeze()).abs().tolist())
        
        avg_error = np.mean(errors)
        error_pct = (avg_error / np.mean(full_dataset.labels)) * 100
        print(f"Epoch {epoch+1}: Avg Error {avg_error:.2f}ms ({error_pct:.1f}%)")

if __name__ == "__main__":
    train_model()
