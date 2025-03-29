import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random

# Define important metrics for scheduling sequence
important_metrics = [
    'bytes_at_production', 'bytes_at_realization', 'bytes_at_root', 'bytes_at_task',
    'inner_parallelism', 'outer_parallelism', 'num_productions', 'num_realizations',
    'num_scalars', 'num_vectors', 'points_computed_total', 'working_set'
]

def get_execution_time(file_path):
    try:
        with open(file_path, 'rb') as f:
            raw_content = f.read()
            content = raw_content.decode('utf-8', errors='replace').replace('\0', '')
            data = json.loads(content)
        
        if 'programming_details' not in data:
            print(f"Error: 'programming_details' key not found in {file_path}")
            return None
        
        schedules = data["scheduling_data"]
        for item in schedules:
            if isinstance(item, dict) and item.get('name') == 'total_execution_time_ms':
                execution_time = item.get('value')
                if execution_time is not None:
                    return float(execution_time)
        
        print(f"Warning: 'total_execution_time_ms' not found in 'Schedules' of {file_path}")
        return schedules[len(schedules)-1]["value"]
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {file_path}: {str(e)}")
        return None
    except UnicodeDecodeError as e:
        print(f"Error: Encoding issue in {file_path}: {str(e)}")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred while processing {file_path}: {str(e)}")
        return None

def extract_features_from_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    execution_time = get_execution_time(file_path)
    if execution_time is None:
        print(f"Warning: No execution time found in {file_path}")
        return None
    
    nodes_features = []
    edges_features = []
    programming_details = data.get("programming_details", None)
    
    if programming_details:
        if 'Nodes' in programming_details:
            for node in programming_details['Nodes']:
                node_feature = {'Name': node.get('Name', '')}
                if 'Details' in node and 'Op histogram' in node['Details']:
                    op_hist = node['Details']['Op histogram']
                    for op_line in op_hist:
                        parts = op_line.strip().split(':')
                        if len(parts) == 2:
                            op_name = parts[0].strip()
                            op_count = int(parts[1].strip())
                            node_feature[f'op_{op_name.lower()}'] = op_count
                nodes_features.append(node_feature)
        
        if 'Edges' in programming_details:
            for edge in programming_details['Edges']:
                edge_feature = {
                    'From': edge.get('From', ''),
                    'To': edge.get('To', ''),
                    'Name': edge.get('Name', '')
                }
                edges_features.append(edge_feature)
    
    scheduling_features = []
    scheduling_data = data.get("scheduling_data", None)
    if not scheduling_data and programming_details and 'Schedules' in programming_details:
        scheduling_data = programming_details['Schedules']
    
    if scheduling_data:
        for sched in scheduling_data:
            sched_feature = {'Name': sched.get('Name', '')}
            if 'Details' in sched and 'scheduling_feature' in sched['Details']:
                sf = sched['Details']['scheduling_feature']
                sched_feature.update(sf)
            scheduling_features.append(sched_feature)
    
    # Create scheduling sequence
    scheduling_sequence = []
    for sf in scheduling_features:
        seq_vector = [float(sf.get(metric, 0.0)) for metric in important_metrics]
        scheduling_sequence.append(seq_vector)
    if not scheduling_sequence:  # Ensure at least one time step
        scheduling_sequence = [[0.0] * len(important_metrics)]
    
    # Calculate operation counts for scalar features
    op_counts = {}
    for node in nodes_features:
        for key, value in node.items():
            if key.startswith('op_'):
                op_counts[key] = op_counts.get(key, 0) + value
    
    # Define scalar features
    scalar_features = {
        'nodes_count': len(nodes_features),
        'edges_count': len(edges_features),
        'node_edge_ratio': len(nodes_features) / len(edges_features) if len(edges_features) > 0 else 0,
        'total_ops': sum(op_counts.values()),
        'op_diversity': len(op_counts) / len(nodes_features) if len(nodes_features) > 0 else 0
    }
    
    return {
        'scheduling_sequence': scheduling_sequence,
        'scalar_features': scalar_features,
        'execution_time': execution_time
    }

def process_directory(directory_path):
    all_features = []
    file_names = []
    json_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.json')])
    
    for filename in json_files:
        file_path = os.path.join(directory_path, filename)
        features = extract_features_from_file(file_path)
        if features is not None:
            all_features.append(features)
            file_names.append(filename)
    
    return all_features, file_names

def process_main_directory(main_dir):
    all_features = []
    all_file_names = []
    subdirs = sorted([d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))])
    
    if len(subdirs) < 1:
        raise ValueError(f"Expected at least 1 subdirectory in {main_dir}, found {len(subdirs)}")
    
    for subdir in subdirs:
        subdir_path = os.path.join(main_dir, subdir)
        features, file_names = process_directory(subdir_path)
        if not features:
            print(f"Skipping {subdir} due to no valid data")
            continue
        all_features.extend(features)
        all_file_names.extend([os.path.join(subdir, fname) for fname in file_names])
        print(f"Processed subdir {subdir}: {len(features)} files")
    
    total_files = len(all_features)
    if total_files < 50:
        raise ValueError(f"Expected at least 50 files total, found {total_files}")
    
    combined = list(zip(all_features, all_file_names))
    random.shuffle(combined)
    all_features, all_file_names = zip(*combined)
    
    test_size = 50
    train_features = all_features[:-test_size]
    test_features = all_features[-test_size:]
    train_file_names = all_file_names[:-test_size]
    test_file_names = all_file_names[-test_size:]
    
    print(f"Total files: {total_files}")
    print(f"Training files: {len(train_features)}")
    print(f"Testing files: {len(test_features)}")
    
    return train_features, test_features, list(test_file_names)

def prepare_data_for_model(train_features, test_features):
    # Extract sequences
    train_sequences = [torch.FloatTensor(f['scheduling_sequence']) for f in train_features]
    test_sequences = [torch.FloatTensor(f['scheduling_sequence']) for f in test_features]
    
    # Pad sequences
    train_sequences_padded = pad_sequence(train_sequences, batch_first=True)
    test_sequences_padded = pad_sequence(test_sequences, batch_first=True)
    
    # Extract scalar features
    train_scalar_df = pd.DataFrame([f['scalar_features'] for f in train_features])
    test_scalar_df = pd.DataFrame([f['scalar_features'] for f in test_features])
    
    # Extract execution times
    y_train = np.array([f['execution_time'] for f in train_features]).reshape(-1, 1)
    y_test = np.array([f['execution_time'] for f in test_features]).reshape(-1, 1)
    
    # Normalize scalar features and targets
    scaler_X_scalar = StandardScaler()
    scaler_y = StandardScaler()
    
    train_scalar_scaled = scaler_X_scalar.fit_transform(train_scalar_df)
    test_scalar_scaled = scaler_X_scalar.transform(test_scalar_df)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Create tensors
    train_scalar_tensor = torch.FloatTensor(train_scalar_scaled)
    test_scalar_tensor = torch.FloatTensor(test_scalar_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    print(f"Sequence input size: {train_sequences_padded.shape[2]}")
    print(f"Scalar input size: {train_scalar_tensor.shape[1]}")
    
    return (train_sequences_padded, train_scalar_tensor, y_train_tensor,
            test_sequences_padded, test_scalar_tensor, y_test_tensor,
            scaler_y, train_sequences_padded.shape[2], train_scalar_tensor.shape[1])

class RecursiveLSTMModel(nn.Module):
    def __init__(self, seq_input_size, scalar_input_size, hidden_size=128, output_size=1, dropout_rate=0.5):
        super(RecursiveLSTMModel, self).__init__()
        
        # LSTM for processing scheduling sequence
        self.lstm = nn.LSTM(seq_input_size, hidden_size, batch_first=True)
        
        # Fully connected layers after concatenation
        self.fc1 = nn.Linear(hidden_size + scalar_input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.output_layer = nn.Linear(32, output_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, seq_input, scalar_input):
        # Process sequence with LSTM
        lstm_out, _ = self.lstm(seq_input)
        lstm_out = lstm_out[:, -1, :]  # Use the last hidden state
        
        # Concatenate with scalar features
        combined = torch.cat((lstm_out, scalar_input), dim=1)
        
        # Fully connected layers
        x = self.fc1(combined)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.output_layer(x)
        
        return output

def create_data_loaders(train_sequences, train_scalar, y_train, test_sequences, test_scalar, y_test, batch_size=64):
    train_dataset = TensorDataset(train_sequences, train_scalar, y_train)
    test_dataset = TensorDataset(test_sequences, test_scalar, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=150, patience=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for seq_inputs, scalar_inputs, targets in train_loader:
            seq_inputs, scalar_inputs, targets = seq_inputs.to(device), scalar_inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(seq_inputs, scalar_inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * seq_inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq_inputs, scalar_inputs, targets in test_loader:
                seq_inputs, scalar_inputs, targets = seq_inputs.to(device), scalar_inputs.to(device), targets.to(device)
                outputs = model(seq_inputs, scalar_inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * seq_inputs.size(0)
        
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
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
    
    return train_losses, val_losses

def evaluate_model(model, X_test_seq, X_test_scalar, y_test, y_scaler, file_names_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    X_test_seq, X_test_scalar = X_test_seq.to(device), X_test_scalar.to(device)
    with torch.no_grad():
        y_pred_scaled = model(X_test_seq, X_test_scalar)
    
    y_pred_scaled = y_pred_scaled.cpu().numpy()
    y_test = y_test.cpu().numpy()
    
    y_test_transformed = y_scaler.inverse_transform(y_test)
    y_pred_transformed = y_scaler.inverse_transform(y_pred_scaled)
    
    results_by_subfolder = {}
    for i, file_path in enumerate(file_names_test):
        subfolder = file_path.split('/')[0]
        if subfolder not in results_by_subfolder:
            results_by_subfolder[subfolder] = []
        
        results_by_subfolder[subfolder].append({
            'file': file_path,
            'actual': y_test_transformed[i][0],
            'predicted': y_pred_transformed[i][0],
            'error_percentage': abs(y_test_transformed[i][0] - y_pred_transformed[i][0]) / y_test_transformed[i][0] * 100 if y_test_transformed[i][0] > 0 else 0
        })
    
    for subfolder, results in results_by_subfolder.items():
        print(f"\nResults for {subfolder}:")
        for result in results:
            print(f"File: {result['file']}")
            print(f"  Actual execution time: {result['actual']:.2f} ms")
            print(f"  Predicted execution time: {result['predicted']:.2f} ms")
            print(f"  Error percentage: {result['error_percentage']:.2f}%")
    
    mse = np.mean((y_test_transformed - y_pred_transformed) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_transformed - y_pred_transformed))
    mape = np.mean(np.abs((y_test_transformed - y_pred_transformed) / (y_test_transformed + 1e-8))) * 100
    
    print("\nOverall Model Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    return y_test_transformed, y_pred_transformed

def main(main_dir):
    print(f"Processing main directory: {main_dir}")
    train_features, test_features, test_file_names = process_main_directory(main_dir)
    
    print(f"Total training samples: {len(train_features)} (randomly selected)")
    print(f"Total test samples: {len(test_features)} (50 randomly selected)")
    
    if len(train_features) == 0 or len(test_features) == 0:
        print("Error: No valid training or test data found")
        return None
    
    # Prepare data
    (train_sequences, train_scalar, y_train,
     test_sequences, test_scalar, y_test,
     y_scaler, seq_input_size, scalar_input_size) = prepare_data_for_model(train_features, test_features)
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        train_sequences, train_scalar, y_train,
        test_sequences, test_scalar, y_test,
        batch_size=64
    )
    
    # Initialize model
    model = RecursiveLSTMModel(
        seq_input_size=seq_input_size,
        scalar_input_size=scalar_input_size,
        hidden_size=128,
        output_size=1,
        dropout_rate=0.5
    )
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Train model
    print("Building and training Recursive LSTM model...")
    train_losses, val_losses = train_model(
        model, train_loader, test_loader,
        criterion, optimizer,
        num_epochs=150, patience=20
    )
    
    # Evaluate model
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(
        model, test_sequences, test_scalar, y_test,
        y_scaler, test_file_names
    )
    
    return model, y_scaler, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "synthetic_data"
    random.seed(42)
    model, y_scaler, y_test_actual, y_pred_actual = main(main_dir)
