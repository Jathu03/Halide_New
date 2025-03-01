import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Step 1: Load JSON Files from Directory
def load_json_files(directory):
    data = []
    program_names = []
    
    for program_folder in os.listdir(directory):
        program_path = os.path.join(directory, program_folder)
        if os.path.isdir(program_path):
            program_data = []
            for json_file in os.listdir(program_path):
                if json_file.endswith('.json'):
                    file_path = os.path.join(program_path, json_file)
                    try:
                        with open(file_path, 'r') as f:
                            json_data = json.load(f)
                            program_data.append(json_data)
                    except json.JSONDecodeError as e:
                        print(f"Skipping {file_path}: Invalid JSON ({e})")
                    except Exception as e:
                        print(f"Skipping {file_path}: Error ({e})")
            if program_data:
                data.append(program_data)
                program_names.append(program_folder)
            else:
                print(f"Warning: No valid JSON files loaded from {program_folder}")
    
    if not data:
        raise ValueError("No valid data loaded from Output_Programs folder")
    return data, program_names

# Step 2: Enhanced Feature Extraction with Better Execution Time Detection
def extract_features(json_data, debug=False):
    features = []
    
    # Extract from Edges
    edges = json_data.get('programming_details', {}).get('Edges', [])
    num_edges = len(edges)
    footprint_sizes = [len(edge.get('Details', {}).get('Footprint', [])) for edge in edges if 'Details' in edge]
    jacobian_sizes = [len(edge.get('Details', {}).get('Load Jacobians', [])) for edge in edges if 'Details' in edge]
    
    features.extend([
        num_edges,
        np.mean(footprint_sizes) if footprint_sizes else 0,
        np.mean(jacobian_sizes) if jacobian_sizes else 0
    ])
    
    # Extract from Nodes
    nodes = json_data.get('programming_details', {}).get('Nodes', [])
    num_nodes = len(nodes)
    
    # More robust memory_patterns extraction
    memory_patterns = []
    for node in nodes:
        if 'Details' in node and 'Memory access patterns' in node['Details']:
            patterns = node['Details']['Memory access patterns']
            if patterns and isinstance(patterns, list) and patterns[0]:
                try:
                    # Try different pattern formats
                    pattern = patterns[0]
                    if isinstance(pattern, str):
                        # Format: "pattern 1 2 3 4"
                        values = pattern.split()[1:]
                        memory_patterns.append(sum(map(int, values)))
                    elif isinstance(pattern, dict) and 'pattern' in pattern:
                        # Format: {"pattern": "1 2 3 4"}
                        values = pattern['pattern'].split()
                        memory_patterns.append(sum(map(int, values)))
                except (ValueError, IndexError, KeyError):
                    continue
    
    # More robust op_histogram extraction
    op_histogram = []
    for node in nodes:
        if 'Details' in node and 'Op histogram' in node['Details']:
            ops = node['Details']['Op histogram']
            if ops and isinstance(ops, list):
                try:
                    # Try different op formats
                    for op in ops:
                        if isinstance(op, str) and len(op.split()) > 0:
                            # Format: "op_name 123"
                            op_histogram.append(int(op.split()[-1]))
                        elif isinstance(op, dict) and 'count' in op:
                            # Format: {"op": "op_name", "count": 123}
                            op_histogram.append(int(op['count']))
                except (ValueError, IndexError, KeyError):
                    continue
    
    features.extend([
        num_nodes,
        np.mean(memory_patterns) if memory_patterns else 0,
        np.mean(op_histogram) if op_histogram else 0
    ])
    
    # Add fallback features
    # If we have too few features, extend with zeros
    while len(features) < 6:  # Assuming we need at least 6 features
        features.append(0)
    
    # ENHANCED EXECUTION TIME EXTRACTION
    # Try multiple possible locations for execution time
    execution_time = None
    
    # Debug full structure
    if debug:
        print(f"JSON structure keys: {json_data.keys()}")
        if 'schedule_feature' in json_data:
            print(f"schedule_feature type: {type(json_data['schedule_feature'])}")
            print(f"schedule_feature content: {json_data['schedule_feature']}")
    
    # Method 1: Try the original structure (list of dicts in schedule_feature)
    if 'schedule_feature' in json_data and isinstance(json_data['schedule_feature'], list):
        for item in json_data['schedule_feature']:
            if isinstance(item, dict):
                if item.get('name') == 'total_execution_time_ms':
                    execution_time = float(item.get('value', 0))
                    if debug:
                        print(f"Found execution time in schedule_feature: {execution_time}")
                    break
    
    # Method 2: Check if schedule_feature is a dict
    if execution_time is None and 'schedule_feature' in json_data and isinstance(json_data['schedule_feature'], dict):
        if 'total_execution_time_ms' in json_data['schedule_feature']:
            execution_time = float(json_data['schedule_feature']['total_execution_time_ms'])
            if debug:
                print(f"Found execution time in schedule_feature dict: {execution_time}")
    
    # Method 3: Look for execution_time directly
    if execution_time is None:
        # Try different key names that might contain execution time
        for key in ['total_execution_time_ms', 'execution_time_ms', 'execution_time', 'runtime_ms', 'runtime']:
            if key in json_data:
                execution_time = float(json_data[key])
                if debug:
                    print(f"Found execution time at key '{key}': {execution_time}")
                break
    
    # Method 4: Look in performance_metrics or similar fields
    if execution_time is None:
        for metrics_key in ['performance_metrics', 'metrics', 'details', 'Details', 'stats', 'timing']:
            if metrics_key in json_data and isinstance(json_data[metrics_key], dict):
                for time_key in ['total_execution_time_ms', 'execution_time_ms', 'time_ms', 'runtime_ms']:
                    if time_key in json_data[metrics_key]:
                        execution_time = float(json_data[metrics_key][time_key])
                        if debug:
                            print(f"Found execution time in {metrics_key}.{time_key}: {execution_time}")
                        break
            if execution_time is not None:
                break
    
    # Method 5: Look in nested programming_details
    if execution_time is None and 'programming_details' in json_data:
        details = json_data['programming_details']
        if isinstance(details, dict):
            for time_key in ['total_execution_time_ms', 'execution_time_ms', 'time_ms', 'runtime_ms']:
                if time_key in details:
                    execution_time = float(details[time_key])
                    if debug:
                        print(f"Found execution time in programming_details.{time_key}: {execution_time}")
                    break
    
    # Fallback: If no execution time is found, generate synthetic data for demonstration
    # In a real application, you would instead return None and skip this data point
    if execution_time is None:
        # For debug builds or if asked to use synthetic data, generate a value
        use_synthetic = os.environ.get('USE_SYNTHETIC_DATA', 'False').lower() in ('true', '1', 't')
        if debug or use_synthetic:
            # Generate synthetic execution time based on features
            # More edges, nodes, etc. would generally mean longer execution time
            synthetic_time = (num_edges * 0.5 + num_nodes * 0.3) * (1 + np.random.random() * 0.5)
            execution_time = max(0.1, synthetic_time)  # Ensure positive value
            print(f"Using synthetic execution time: {execution_time:.2f} ms")
        else:
            if debug:
                print("Warning: No execution time found in JSON")
            return np.array(features, dtype=float), None
    
    return np.array(features, dtype=float), execution_time

# Step 3: Prepare Data for LSTM with Better Error Handling
def prepare_lstm_data(data):
    X, y = [], []
    feature_dim = None
    
    # First pass to determine feature dimension
    for program_data in data:
        for schedule in program_data:
            features, _ = extract_features(schedule)
            if feature_dim is None:
                feature_dim = len(features)
            elif len(features) > feature_dim:
                feature_dim = len(features)
    
    if feature_dim is None:
        raise ValueError("Could not determine feature dimension from data")
    
    print(f"Using feature dimension: {feature_dim}")
    
    # Second pass to extract features and targets
    for i, program_data in enumerate(data):
        program_X, program_y = [], []
        schedules_with_time = 0
        
        for j, schedule in enumerate(program_data):
            # Enable debug for the first schedule of each program
            debug_this = (j == 0)
            features, exec_time = extract_features(schedule, debug=debug_this)
            
            if exec_time is not None:
                schedules_with_time += 1
                # Ensure consistent feature dimension
                if len(features) < feature_dim:
                    features = np.pad(features, (0, feature_dim - len(features)), 'constant')
                elif len(features) > feature_dim:
                    features = features[:feature_dim]
                
                program_X.append(features)
                program_y.append(exec_time)
        
        if schedules_with_time > 0:
            X.append(program_X)
            y.append(program_y)
            print(f"Program {i}: Added {schedules_with_time} schedules with execution times")
        else:
            print(f"Warning: No valid schedules with execution time in program {i}")
    
    if not X or not y:
        # If no real data, create synthetic data for demonstration
        print("WARNING: No valid sequences with execution times. Creating synthetic data for demonstration.")
        
        # Generate synthetic data
        n_programs = 10
        max_schedules = 20
        X = []
        y = []
        
        for i in range(n_programs):
            n_schedules = np.random.randint(5, max_schedules)
            program_X = np.random.rand(n_schedules, feature_dim)
            # Create synthetic target values correlated with features
            base_time = np.random.uniform(10, 100)
            program_y = base_time + np.sum(program_X, axis=1) * np.random.uniform(5, 10)
            
            X.append(program_X)
            y.append(program_y)
            print(f"Generated synthetic program {i} with {n_schedules} schedules")
        
        print("IMPORTANT: Using synthetic data for demonstration. Results will not be meaningful.")
    
    # Convert to numpy arrays
    X = np.array(X, dtype=object)  # Allow ragged arrays temporarily
    y = np.array(y, dtype=object)
    
    # Pad sequences to the same length (max num_schedules)
    max_seq_len = max(len(seq) for seq in X)
    X_padded = np.zeros((len(X), max_seq_len, feature_dim))
    y_padded = np.zeros((len(y), max_seq_len))
    
    for i in range(len(X)):
        seq_len = len(X[i])
        X_padded[i, :seq_len, :] = X[i]
        y_padded[i, :seq_len] = y[i]
    
    # Debug target values
    print(f"Target values range: min={y_padded.min()}, max={y_padded.max()}, mean={y_padded.mean()}")
    
    # Normalize features
    scaler_X = MinMaxScaler()
    X_reshaped = X_padded.reshape(-1, X_padded.shape[-1])
    X_normalized = scaler_X.fit_transform(X_reshaped).reshape(X_padded.shape)
    
    # Normalize target
    scaler_y = MinMaxScaler()
    y_reshaped = y_padded.reshape(-1, 1)
    y_normalized = scaler_y.fit_transform(y_reshaped).reshape(y_padded.shape)
    
    return X_normalized, y_normalized, scaler_X, scaler_y

# Step 4: Custom Dataset for PyTorch
class HalideDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Step 5: Define LSTM Model in PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out[:, -1, :])  # Take the last timestep
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Step 6: Training and Evaluation
def train_model(model, train_loader, val_loader, epochs=50, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 10
    counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch[:, -1].unsqueeze(1))  # Predict last timestep
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch[:, -1].unsqueeze(1)).item()
        
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_lstm_model.pt')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_lstm_model.pt'))
    return model

def evaluate_model(model, test_loader, scaler_y, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.append(outputs.cpu().numpy())
            actuals.append(y_batch[:, -1].numpy())  # Last timestep
    
    y_pred = np.concatenate(predictions)
    y_test = np.concatenate(actuals)
    
    # Reshape and denormalize
    y_pred_rescaled = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = np.mean((y_test_rescaled - y_pred_rescaled) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_rescaled - y_pred_rescaled))
    mape = np.mean(np.abs((y_test_rescaled - y_pred_rescaled) / (y_test_rescaled + 1e-8))) * 100
    
    print(f"Test MSE: {mse:.6f}")
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    print(f"Test MAPE: {mape:.2f}%")
    
    return y_test_rescaled, y_pred_rescaled

# Main Execution
def predict_halide_speedup(data_dir, use_synthetic=False):
    if use_synthetic:
        os.environ['USE_SYNTHETIC_DATA'] = 'True'
    
    # Load and prepare data
    data, program_names = load_json_files(data_dir)
    print(f"Loaded data from {len(program_names)} programs")
    
    # Print first few programs to debug
    print(f"Program names: {program_names[:5]}")
    
    # Try to prepare data
    try:
        X, y, scaler_X, scaler_y = prepare_lstm_data(data)
        print(f"X shape: {X.shape}, y shape: {y.shape}")
    except Exception as e:
        print(f"Error preparing data: {e}")
        print("Trying with synthetic data...")
        os.environ['USE_SYNTHETIC_DATA'] = 'True'
        X, y, scaler_X, scaler_y = prepare_lstm_data(data)
        print(f"X shape with synthetic data: {X.shape}, y shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 60% train, 20% val, 20% test
    
    # Create datasets and loaders
    train_dataset = HalideDataset(X_train, y_train)
    val_dataset = HalideDataset(X_val, y_val)
    test_dataset = HalideDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    input_size = X.shape[2]  # feature_dim
    model = LSTMModel(input_size)
    
    # Train model
    model = train_model(model, train_loader, val_loader, epochs=50)
    
    # Evaluate model
    y_test_rescaled, y_pred_rescaled = evaluate_model(model, test_loader, scaler_y)
    
    # Print example predictions
    for i in range(min(5, len(y_test_rescaled))):
        print(f"Example {i+1}: True Time: {y_test_rescaled[i]:.2f} ms, Predicted Time: {y_pred_rescaled[i]:.2f} ms")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_dim': input_size
    }, 'lstm_execution_time_predictor.pt')
    
    print("Model saved as 'lstm_execution_time_predictor.pt'")
    
    return model, scaler_X, scaler_y

if __name__ == "__main__":
    # Try with real data first, fall back to synthetic if needed
    predict_halide_speedup(data_dir="Output_Programs", use_synthetic=False)
