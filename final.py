import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

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
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def extract_features_from_file(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        execution_time = get_execution_time(file_path)
        
        if execution_time is None:
            print(f"Warning: No execution time found in {file_path}")
            return None
        
        # Extract basic node features
        nodes_features = []
        edges_features = []
        programming_details = data.get("programming_details", {})
        
        if programming_details:
            if 'Nodes' in programming_details:
                for node in programming_details['Nodes']:
                    node_feature = {}
                    node_feature['Name'] = node.get('Name', '')
                    # Extract operation histogram
                    if 'Details' in node and 'Op histogram' in node['Details']:
                        op_hist = node['Details']['Op histogram']
                        for op_line in op_hist:
                            parts = op_line.strip().split(':')
                            if len(parts) == 2:
                                op_name = parts[0].strip()
                                op_count = int(parts[1].strip())
                                node_feature[f'op_{op_name.lower()}'] = op_count
                    
                    # Extract additional node metrics if available
                    if 'Details' in node:
                        for key, value in node['Details'].items():
                            if key != 'Op histogram' and isinstance(value, (int, float)):
                                node_feature[f'node_{key.lower()}'] = value
                    
                    nodes_features.append(node_feature)
            
            if 'Edges' in programming_details:
                for edge in programming_details['Edges']:
                    edge_feature = {}
                    edge_feature['From'] = edge.get('From', '')
                    edge_feature['To'] = edge.get('To', '')
                    edge_feature['Name'] = edge.get('Name', '')
                    if 'Details' in edge:
                        for key, value in edge['Details'].items():
                            if isinstance(value, (int, float)):
                                edge_feature[f'edge_{key.lower()}'] = value
                    edges_features.append(edge_feature)
        
        # Extract scheduling features
        scheduling_features = []
        scheduling_data = data.get("scheduling_data", None)
        
        if not scheduling_data and 'Schedules' in programming_details:
            scheduling_data = programming_details['Schedules']
        
        if scheduling_data:
            for sched in scheduling_data:
                if not isinstance(sched, dict):
                    continue
                
                sched_feature = {}
                sched_feature['Name'] = sched.get('name', sched.get('Name', ''))
                
                # Extract scheduling details
                details = sched.get('Details', {})
                if isinstance(details, dict) and 'scheduling_feature' in details:
                    sf = details['scheduling_feature']
                    for key, value in sf.items():
                        if isinstance(value, (int, float)):
                            sched_feature[key] = value
                
                # Also extract direct values
                for key, value in sched.items():
                    if key not in ['name', 'Name', 'Details'] and isinstance(value, (int, float)):
                        sched_feature[key] = value
                
                scheduling_features.append(sched_feature)
        
        # Combine all features
        features = {
            'execution_time': execution_time,
            'nodes_count': len(nodes_features),
            'edges_count': len(edges_features),
            'scheduling_count': len(scheduling_features)
        }
        
        # Aggregate operation counts
        op_counts = {}
        for node in nodes_features:
            for key, value in node.items():
                if key.startswith('op_'):
                    op_counts[key] = op_counts.get(key, 0) + value
        features.update(op_counts)
        
        # Calculate node complexity metrics
        if nodes_features:
            op_types = set()
            for node in nodes_features:
                for key in node.keys():
                    if key.startswith('op_'):
                        op_types.add(key)
            
            features['unique_op_types'] = len(op_types)
            
            # Calculate node connectivity
            node_names = {node.get('Name', '') for node in nodes_features if node.get('Name')}
            connected_nodes = set()
            for edge in edges_features:
                connected_nodes.add(edge.get('From', ''))
                connected_nodes.add(edge.get('To', ''))
            
            features['connected_node_ratio'] = len(connected_nodes & node_names) / len(node_names) if node_names else 0
        
        # Extract and combine important scheduling metrics
        if scheduling_features:
            important_metrics = [
                'bytes_at_production', 'bytes_at_realization', 'bytes_at_root', 'bytes_at_task',
                'inner_parallelism', 'outer_parallelism', 'num_productions', 'num_realizations',
                'num_scalars', 'num_vectors', 'points_computed_total', 'working_set',
                'value', 'max_stack_depth', 'memory_footprint', 'buffer_size'
            ]
            
            for metric in important_metrics:
                # Aggregate metrics from all scheduling features
                metric_values = [sf.get(metric, 0) for sf in scheduling_features if isinstance(sf, dict)]
                if metric_values:
                    features[f'sched_sum_{metric}'] = sum(metric_values)
                    features[f'sched_max_{metric}'] = max(metric_values)
                    features[f'sched_avg_{metric}'] = sum(metric_values) / len(metric_values)
            
            # Calculate advanced scheduling metrics
            total_bytes_at_production = sum(sf.get('bytes_at_production', 0) for sf in scheduling_features if isinstance(sf, dict))
            total_vectors = sum(sf.get('num_vectors', 0) for sf in scheduling_features if isinstance(sf, dict))
            total_parallelism = sum((sf.get('inner_parallelism', 0) or 1) * (sf.get('outer_parallelism', 1) or 1) 
                                  for sf in scheduling_features if isinstance(sf, dict))
            
            features['total_bytes_at_production'] = total_bytes_at_production
            features['total_vectors'] = total_vectors
            features['total_parallelism'] = total_parallelism
            
            # Memory efficiency metrics
            if total_bytes_at_production > 0 and total_vectors > 0:
                features['bytes_per_vector'] = total_bytes_at_production / total_vectors
            
            # Derived performance indicators
            working_sets = [sf.get('working_set', 0) for sf in scheduling_features if isinstance(sf, dict) and 'working_set' in sf]
            if working_sets:
                features['max_working_set'] = max(working_sets)
                features['total_working_set'] = sum(working_sets)
        
        return features
    
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

def process_directory(directory_path, train_ratio=0.9):
    """Process all JSON files in a directory and split into train and test sets based on ratio."""
    all_features = []
    file_names = []
    
    # Get all JSON files in the directory
    json_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.json')])
    
    if len(json_files) < 5:
        print(f"Warning: Only {len(json_files)} files found in {directory_path}")
    
    # Process each file and extract features
    for filename in json_files:
        file_path = os.path.join(directory_path, filename)
        features = extract_features_from_file(file_path)
        if features is not None:
            all_features.append(features)
            file_names.append(filename)
    
    # Ensure we have enough files to work with
    if len(all_features) < 5:
        print(f"Warning: Only {len(all_features)} valid files found in {directory_path}")
        return None, None, None
    
    # Split into training and testing
    train_size = int(len(all_features) * train_ratio)
    train_features = all_features[:train_size]
    test_features = all_features[train_size:]
    test_file_names = file_names[train_size:]
    
    return train_features, test_features, test_file_names

def process_main_directory(main_dir, train_ratio=0.9):
    """Process all subdirectories, splitting each into train/test sets."""
    all_train_features = []
    all_test_features = []
    all_test_file_names = []
    
    # Get all subdirectories
    subdirs = sorted([d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))])
    
    if len(subdirs) < 1:
        raise ValueError(f"Expected at least 1 subdirectory in {main_dir}, found {len(subdirs)}")
    
    # Process each subdirectory
    for subdir in subdirs:
        subdir_path = os.path.join(main_dir, subdir)
        train_features, test_features, test_file_names = process_directory(subdir_path, train_ratio)
        
        if train_features is None or test_features is None:
            print(f"Skipping {subdir} due to insufficient data")
            continue
        
        all_train_features.extend(train_features)
        all_test_features.extend(test_features)
        all_test_file_names.extend([os.path.join(subdir, fname) for fname in test_file_names])
        
        print(f"Processed subdir {subdir}: {len(train_features)} training files, {len(test_features)} test files")
    
    return all_train_features, all_test_features, all_test_file_names

def clean_and_prepare_features(features_list):
    """Clean and prepare features for model training."""
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(features_list)
    
    # Handle missing values
    df = df.fillna(0)
    
    # Check for target variable
    if 'execution_time' not in df.columns:
        raise ValueError("Target variable 'execution_time' not found in features")
    
    # Log transform of execution time if it's skewed
    if (df['execution_time'] > 0).all():
        skewness = df['execution_time'].skew()
        if abs(skewness) > 1:
            df['execution_time'] = np.log1p(df['execution_time'])
            print(f"Applied log transform to execution_time (skewness: {skewness:.2f})")
    
    # Convert any string columns to numeric if possible
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                # Drop columns that can't be converted to numeric
                df = df.drop(col, axis=1)
    
    # Create feature ratios that might be predictive
    if 'nodes_count' in df.columns and 'edges_count' in df.columns and df['nodes_count'].sum() > 0:
        df['edge_to_node_ratio'] = df['edges_count'] / df['nodes_count'].clip(lower=1)
    
    if 'total_bytes_at_production' in df.columns and 'nodes_count' in df.columns and df['nodes_count'].sum() > 0:
        df['bytes_per_node'] = df['total_bytes_at_production'] / df['nodes_count'].clip(lower=1)
    
    return df

def prepare_data_for_model(train_features, test_features):
    """Prepare the training and testing data for the model."""
    # Clean and prepare features
    train_df = clean_and_prepare_features(train_features)
    # Apply same transformations to test set
    test_df = clean_and_prepare_features(test_features)
    
    # Ensure test_df has the same columns as train_df
    for col in train_df.columns:
        if col not in test_df.columns:
            test_df[col] = 0
    
    # Keep only columns that are in both dataframes
    common_cols = set(train_df.columns) & set(test_df.columns)
    train_df = train_df[[col for col in train_df.columns if col in common_cols]]
    test_df = test_df[[col for col in train_df.columns]]
    
    # Split features and target
    X_train = train_df.drop('execution_time', axis=1)
    y_train = train_df['execution_time'].values.reshape(-1, 1)
    X_test = test_df.drop('execution_time', axis=1)
    y_test = test_df['execution_time'].values.reshape(-1, 1)
    
    # Feature scaling
    scaler_X = RobustScaler()  # More robust to outliers than StandardScaler
    scaler_y = RobustScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)  # Add sequence dimension
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    # Save feature names for later interpretation
    feature_names = X_train.columns.tolist()
    
    return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
            scaler_y, X_train_scaled.shape[1], feature_names, y_train, y_test)

class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, dropout_rate=0.3):
        super(EnhancedLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True, bidirectional=True)
        self.bn1 = nn.BatchNorm1d(hidden_size1 * 2)  # For bidirectional
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.lstm2 = nn.LSTM(hidden_size1 * 2, hidden_size2, batch_first=True, bidirectional=True)
        self.bn2 = nn.BatchNorm1d(hidden_size2 * 2)  # For bidirectional
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size2 * 2, 1),
            nn.Tanh(),
            nn.Softmax(dim=1)
        )
        
        self.fc1 = nn.Linear(hidden_size2 * 2, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.relu = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        # First LSTM layer
        lstm_out1, _ = self.lstm1(x)
        # Apply batch normalization (need to reshape for BatchNorm1d)
        lstm_out1 = lstm_out1.reshape(-1, lstm_out1.size(2))
        lstm_out1 = self.bn1(lstm_out1)
        lstm_out1 = lstm_out1.reshape(x.size(0), x.size(1), -1)
        lstm_out1 = self.dropout1(lstm_out1)
        
        # Second LSTM layer
        lstm_out2, _ = self.lstm2(lstm_out1)
        # Apply batch normalization
        reshaped_lstm_out2 = lstm_out2.reshape(-1, lstm_out2.size(2))
        reshaped_lstm_out2 = self.bn2(reshaped_lstm_out2)
        lstm_out2 = reshaped_lstm_out2.reshape(x.size(0), x.size(1), -1)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out2)
        context_vector = torch.sum(attention_weights * lstm_out2, dim=1)
        
        # Fully connected layers
        fc1_out = self.fc1(context_vector)
        fc1_out = self.bn3(fc1_out)
        fc1_out = self.relu(fc1_out)
        fc1_out = self.dropout3(fc1_out)
        output = self.fc2(fc1_out)
        
        return output

class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.3):
        super(CNNLSTMModel, self).__init__()
        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        
        # LSTM layer for sequential processing
        self.lstm = nn.LSTM(64, hidden_size, batch_first=True, bidirectional=True)
        self.bn3 = nn.BatchNorm1d(hidden_size * 2)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.relu3 = nn.LeakyReLU(0.1)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        # CNN feature extraction (need to reshape for Conv1d)
        # Conv1d expects [batch, channels, length]
        batch_size, seq_len, features = x.size()
        x = x.reshape(batch_size, 1, -1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Reshape for LSTM [batch, sequence, features]
        x = x.permute(0, 2, 1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.bn3(lstm_out)
        lstm_out = self.dropout(lstm_out)
        
        # Fully connected layers
        fc1_out = self.fc1(lstm_out)
        fc1_out = self.bn4(fc1_out)
        fc1_out = self.relu3(fc1_out)
        fc1_out = self.dropout2(fc1_out)
        output = self.fc2(fc1_out)
        
        return output

class EnsembleModel(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(EnsembleModel, self).__init__()
        self.lstm_model = EnhancedLSTMModel(input_size, hidden_size)
        self.cnn_lstm_model = CNNLSTMModel(input_size, hidden_size)
        
        # Weight for combining models (learnable parameter)
        self.weight = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
    
    def forward(self, x):
        lstm_out = self.lstm_model(x)
        cnn_lstm_out = self.cnn_lstm_model(x)
        
        # Weighted combination
        weight = torch.sigmoid(self.weight)  # Between 0 and 1
        ensemble_out = weight * lstm_out + (1 - weight) * cnn_lstm_out
        
        return ensemble_out

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=16):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, 
                num_epochs=200, patience=20, clip_value=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        
        # Learning rate scheduling based on validation loss
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
            print(f"New best model saved! (Val Loss: {val_loss:.6f})")
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs.')
            model.load_state_dict(best_model_state)
            break
    
    if best_model_state is not None and epochs_no_improve > 0:
        model.load_state_dict(best_model_state)
        print("Restored best model from checkpoint.")
    
    return train_losses, val_losses

def evaluate_model(model, X_test, y_test, y_scaler, file_names_test, original_y_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    X_test = X_test.to(device)
    with torch.no_grad():
        y_pred_scaled = model(X_test)
    
    y_pred_scaled = y_pred_scaled.cpu().numpy()
    y_test_scaled = y_test.cpu().numpy()
    
    # Inverse transform predictions - handle log transform if applied
    y_test_actual = original_y_test  # This is the original non-transformed target
    y_pred_actual = y_scaler.inverse_transform(y_pred_scaled)
    
    # If log transform was applied, we need to expm1
    if np.mean(y_pred_actual) < np.mean(original_y_test) * 0.1:  # Crude check for log transform
        y_pred_actual = np.expm1(y_pred_actual)
        print("Applied inverse log transform to predictions")
    
    # Group results by subfolder
    results_by_subfolder = {}
    for i, file_path in enumerate(file_names_test):
        subfolder = file_path.split('/')[0]
        if subfolder not in results_by_subfolder:
            results_by_subfolder[subfolder] = []
        
        results_by_subfolder[subfolder].append({
            'file': file_path,
            'actual': y_test_actual[i][0],
            'predicted': y_pred_actual[i][0],
            'error_pct': abs(y_test_actual[i][0] - y_pred_actual[i][0]) / y_test_actual[i][0] * 100 if y_test_actual[i][0] > 0 else 0
        })
    
    # Print results for each subfolder
    overall_errors = []
    for subfolder, results in results_by_subfolder.items():
        print(f"\nResults for {subfolder}:")
        subfolder_errors = []
        for result in results:
            print(f"File: {result['file']}")
            print(f"  Actual: {result['actual']:.2f} ms, Predicted: {result['predicted']:.2f} ms, Error: {result['error_pct']:.2f}%")
            subfolder_errors.append(result['error_pct'])
            overall_errors.append(result['error_pct'])
        
        avg_error = np.mean(subfolder_errors)
        print(f"  Average error for {subfolder}: {avg_error:.2f}%")
    
    # Calculate overall metrics
    mse = np.mean((y_test_actual - y_pred_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_actual - y_pred_actual))
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / np.maximum(y_test_actual, 1e-8))) * 100
    
    print("\nOverall Model Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Average Absolute Percentage Error: {np.mean(overall_errors):.2f}%")
    
    return y_test_actual, y_pred_actual

def main(main_dir, model_type='ensemble', train_ratio=0.9):
    print(f"Processing main directory: {main_dir}")
    train_features, test_features, test_file_names = process_main_directory(main_dir, train_ratio)
    
    print(f"Total training samples: {len(train_features)}")
    print(f"Total test samples: {len(test_features)}")
    
    if len(train_features) == 0 or len(test_features) == 0:
        print("Error: No valid training or test data found")
        return None
    
    # Prepare data for model
    data = prepare_data_for_model(train_features, test_features)
    X_train, y_train, X_test, y_test, y_scaler, input_size, feature_names, orig_y_train, orig_y_test = data
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=16)
    
    # Select model type
    if model_type == 'lstm':
        model = EnhancedLSTMModel(input_size=input_size, hidden_size1=128, hidden_size2=64)
    elif model_type == 'cnn_lstm':
        model = CNNL
