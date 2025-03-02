import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_execution_time(file_path):
    try:
        with open(file_path, 'rb') as f:
            raw_content = f.read()
            content = raw_content.decode('utf-8', errors='replace').replace('\0', '')
            data = json.loads(content)
        schedules = data["scheduling_data"]
        for item in schedules:
            if isinstance(item, dict) and item.get('name') == 'total_execution_time_ms':
                execution_time = item.get('value')
                if execution_time is not None:
                    return float(execution_time)
        print(f"Warning: 'total_execution_time_ms' not found in 'Schedules' of {file_path}")
        return schedules[-1]["value"]
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def extract_optimized_features(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        execution_time = get_execution_time(file_path)
        if execution_time is None or execution_time < 0:
            print(f"Invalid execution time in {file_path}: {execution_time}")
            return None
        
        nodes = data["programming_details"]["Nodes"]
        edges = data["programming_details"]["Edges"]
        scheduling = data["scheduling_data"]
        
        node_dict = {node["Name"]: node["Details"] for node in nodes}
        sched_dict = {item["Name"]: item["Details"]["scheduling_feature"] for item in scheduling if "Name" in item and "Details" in item}
        
        # Computation features (operation counts)
        op_counts = []
        for node_name, details in node_dict.items():
            op_hist = details.get("Op histogram", [])
            ops = {'add': 0, 'mul': 0, 'div': 0, 'min': 0, 'max': 0}
            for entry in op_hist:
                parts = entry.split(':')
                if len(parts) == 2:
                    op_name = parts[0].strip().lower()
                    op_count = int(parts[1].strip().split()[0])
                    if op_name in ops:
                        ops[op_name] = op_count
            op_counts.append([ops['add'], ops['mul'], ops['div'], ops['min'], ops['max']])
        comps_tensor = torch.tensor(op_counts, dtype=torch.float32).mean(dim=0)  # [5]
        
        # Scheduling features (simplified transformation vector)
        sched_vector = [
            sum(1 for sched in sched_dict.values() if sched.get("inner_parallelism", 1.0) > 1.0),
            sum(sched.get("unrolled_loop_extent", 1.0) for sched in sched_dict.values()),
            sum(sched.get("vector_size", 1.0) for sched in sched_dict.values()),
            sum(1 for sched in sched_dict.values() if sched.get("unrolled_loop_extent", 1.0) > 1.0),
            sum(sched.get('bytes_at_production', 0) for sched in sched_dict.values()),
            sum(sched.get('num_vectors', 0) for sched in sched_dict.values())
        ]
        sched_tensor = torch.tensor(sched_vector, dtype=torch.float32)  # [6]
        
        # Structural features
        struct_vector = [
            len(nodes),
            len(edges),
            np.mean([len([e for e in edges if e["To"] == n["Name"]]) for n in nodes]) if nodes else 0,
            sum(1 for e in edges if ".update(" in e["To"])
        ]
        struct_tensor = torch.tensor(struct_vector, dtype=torch.float32)  # [4]
        
        # Combine into a single feature vector
        features = torch.cat([comps_tensor, sched_tensor, struct_tensor])  # [15]
        
        return {"features": features, "execution_time": execution_time}
    
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

def process_main_directory(main_dir):
    train_data = []
    test_data = []
    test_file_names = []
    
    subdirs = sorted([d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))])
    
    if not subdirs:
        raise ValueError(f"No subdirectories found in {main_dir}")
    
    for subdir in subdirs:
        subdir_path = os.path.join(main_dir, subdir)
        all_data = []
        all_file_names = []
        
        for filename in sorted(os.listdir(subdir_path)):
            if filename.endswith('.json'):
                file_path = os.path.join(subdir_path, filename)
                data = extract_optimized_features(file_path)
                if data is not None:
                    all_data.append(data)
                    all_file_names.append(filename)
        
        if len(all_data) != 32:
            print(f"Warning: Expected 32 files in {subdir_path}, found {len(all_data)}")
            continue
        
        train_data.extend(all_data[:30])
        test_data.extend(all_data[30:])
        test_file_names.extend([os.path.join(subdir, fname) for fname in all_file_names[30:]])
        print(f"Processed {len(all_data)} files from {subdir}: 30 for training, 2 for testing")
    
    return train_data, test_data, test_file_names

def prepare_data_for_model(train_data, test_data):
    X_train = torch.stack([item["features"] for item in train_data])  # [N_train, input_size]
    y_train = torch.tensor([item["execution_time"] for item in train_data], dtype=torch.float32).view(-1, 1)
    X_test = torch.stack([item["features"] for item in test_data])    # [N_test, input_size]
    y_test = torch.tensor([item["execution_time"] for item in test_data], dtype=torch.float32).view(-1, 1)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train.numpy())
    y_train_scaled = scaler_y.fit_transform(y_train.numpy())
    X_test_scaled = scaler_X.transform(X_test.numpy())
    y_test_scaled = scaler_y.transform(y_test.numpy())
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)  # [N_train, 1, input_size]
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1)   # [N_test, 1, input_size]
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler_y

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=16):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

class SchedulePredictor(nn.Module):
    def __init__(self, input_size=15, hidden_sizes=[128, 64], lstm_hidden_size=100, output_size=1, device="cpu"):
        super(SchedulePredictor, self).__init__()
        self.device = device
        
        # LSTM to process scheduling features
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, batch_first=True, bidirectional=True)
        self.lstm_dropout = nn.Dropout(0.3)
        
        # Feedforward layers for regression
        layer_sizes = [lstm_hidden_size * 2] + hidden_sizes  # *2 for bidirectional
        self.ff_layers = nn.ModuleList()
        self.ff_dropouts = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.ff_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=True))
            nn.init.xavier_uniform_(self.ff_layers[i].weight)
            self.ff_dropouts.append(nn.Dropout(0.3))
        
        # Output layer
        self.predict = nn.Linear(hidden_sizes[-1], output_size, bias=True)
        nn.init.xavier_uniform_(self.predict.weight)
        
        self.elu = nn.ELU()
        self.leaky_relu = nn.LeakyReLU(0.01)
    
    def forward(self, x):
        # x: [batch_size, seq_len=1, input_size]
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len=1, lstm_hidden_size * 2]
        lstm_out = self.lstm_dropout(lstm_out[:, -1, :])  # [batch_size, lstm_hidden_size * 2]
        
        x = lstm_out
        for i in range(len(self.ff_layers)):
            x = self.ff_layers[i](x)
            x = self.ff_dropouts[i](self.elu(x))
        
        out = self.predict(x)
        return self.leaky_relu(out)  # Ensure non-negative output

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=200, patience=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets.squeeze())
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(test_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            model.load_state_dict(best_model_state)
            break
    
    if best_model_state is not None and epochs_no_improve > 0:
        model.load_state_dict(best_model_state)
    
    return model

def evaluate_model(model, X_test, y_test, y_scaler, file_names_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        X_test = X_test.to(device)
        y_pred_scaled = model(X_test)
    
    y_pred_scaled = y_pred_scaled.cpu().numpy()
    y_test = y_test.cpu().numpy()
    
    y_test_actual = y_scaler.inverse_transform(y_test)
    y_pred_actual = y_scaler.inverse_transform(y_pred_scaled)
    y_pred_actual = np.clip(y_pred_actual, 0, None)
    
    print("\nPredicted vs Actual Execution Times for Test Files:")
    for i, file_name in enumerate(file_names_test):
        print(f"File: {file_name}")
        print(f"  Actual Execution Time: {y_test_actual[i][0]:.2f} ms")
        print(f"  Predicted Execution Time: {y_pred_actual[i][0]:.2f} ms")
    
    return y_test_actual, y_pred_actual

def main(main_dir):
    print(f"Processing main directory: {main_dir}")
    train_data, test_data, test_file_names = process_main_directory(main_dir)
    
    print(f"Total training samples: {len(train_data)}")
    print(f"Total test samples: {len(test_data)}")
    
    if len(train_data) == 0 or len(test_data) == 0:
        print("Error: No valid training or test data found")
        return None, None, None
    
    X_train, y_train, X_test, y_test, y_scaler = prepare_data_for_model(train_data, test_data)
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=16)
    
    model = SchedulePredictor(input_size=15, device="cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    print("Training Schedule Predictor model...")
    model = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler)
    
    print("\nEvaluating the model...")
    y_test_actual, y_pred_actual = evaluate_model(model, X_test, y_test, y_scaler, test_file_names)
    
    return model, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "Output_Programs"
    model, y_test_actual, y_pred_actual = main(main_dir)
    if model is not None:
        print("\nModel training and prediction completed!")
