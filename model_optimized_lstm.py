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
import random
import matplotlib.pyplot as plt

def get_execution_time(file_path):
    try:
        with open(file_path, 'rb') as f:
            raw_content = f.read()
            content = raw_content.decode('utf-8', errors='replace').replace('\0', '')
            data = json.loads(content)
        
        if 'programming_details' not in data:
            print(f"Error: 'programming_details' key not found in {file_path}")
            return None
        
        schedules = data.get("scheduling_data", [])
        for item in schedules:
            if isinstance(item, dict) and item.get('name') == 'total_execution_time_ms':
                execution_time = item.get('value')
                if execution_time is not None:
                    print(f"Extracted execution time for {file_path}: {execution_time} ms")
                    return float(execution_time)
        
        if schedules and isinstance(schedules[-1], dict) and "value" in schedules[-1]:
            execution_time = schedules[-1]["value"]
            print(f"Warning: 'total_execution_time_ms' not found in 'Schedules' of {file_path}, using last schedule value: {execution_time} ms")
            return float(execution_time)
        
        print(f"Error: No valid execution time found in {file_path}")
        return None
    
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
    
    execution_time = np.clip(execution_time, 1.0, 10000.0)
    
    nodes_features = []
    edges_features = []
    programming_details = data.get("programming_details")
    
    if programming_details:
        if 'Nodes' in programming_details:
            for node in programming_details['Nodes']:
                node_feature = {}
                node_feature['Name'] = node.get('Name', '')
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
                edge_feature = {}
                edge_feature['From'] = edge.get('From', '')
                edge_feature['To'] = edge.get('To', '')
                edge_feature['Name'] = edge.get('Name', '')
                edges_features.append(edge_feature)
    
    scheduling_features = []
    scheduling_data = data.get("scheduling_data")
    
    if not scheduling_data and programming_details and 'Schedules' in programming_details:
        scheduling_data = programming_details['Schedules']
    
    if scheduling_data:
        for sched in scheduling_data:
            sched_feature = {}
            sched_feature['Name'] = sched.get('Name', '')
            if 'Details' in sched and 'scheduling_feature' in sched['Details']:
                sf = sched['Details']['scheduling_feature']
                for key, value in sf.items():
                    sched_feature[key] = value
            scheduling_features.append(sched_feature)
    
    features = {
        'execution_time': execution_time,
        'nodes_count': len(nodes_features),
        'edges_count': len(edges_features),
        'scheduling_count': len(scheduling_features)
    }
    
    if len(nodes_features) > 0 and len(edges_features) > 0:
        features['node_edge_ratio'] = len(nodes_features) / len(edges_features)
    else:
        features['node_edge_ratio'] = 0
    
    op_counts = {}
    for node in nodes_features:
        for key, value in node.items():
            if key.startswith('op_'):
                op_counts[key] = op_counts.get(key, 0) + value
    features.update(op_counts)
    
    if scheduling_features:
        important_metrics = [
            'bytes_at_production', 'bytes_at_realization', 'bytes_at_root', 'bytes_at_task',
            'inner_parallelism', 'outer_parallelism', 'num_productions', 'num_realizations',
            'num_scalars', 'num_vectors', 'points_computed_total', 'working_set'
        ]
        if scheduling_features and scheduling_features[0]:
            for metric in important_metrics:
                if metric in scheduling_features[0]:
                    features[f'sched_{metric}'] = scheduling_features[0][metric]
        
        total_bytes_at_production = sum(sf.get('bytes_at_production', 0) for sf in scheduling_features if isinstance(sf, dict))
        total_vectors = sum(sf.get('num_vectors', 0) for sf in scheduling_features if isinstance(sf, dict))
        total_parallelism = sum(sf.get('inner_parallelism', 0) * sf.get('outer_parallelism', 1) for sf in scheduling_features if isinstance(sf, dict))
        points_computed_total = sum(sf.get('points_computed_total', 0) for sf in scheduling_features if isinstance(sf, dict))
        working_set = sum(sf.get('working_set', 0) for sf in scheduling_features if isinstance(sf, dict))
        total_inner_parallelism = sum(sf.get('inner_parallelism', 0) for sf in scheduling_features if isinstance(sf, dict))
        
        comp_efficiency = points_computed_total / max(total_bytes_at_production, 1e-4) if total_bytes_at_production != 0 else 0.0
        bytes_processing_rate = total_bytes_at_production / max(execution_time, 1e-4) if execution_time != 0 else 0.0
        mem_util_ratio = working_set / max(total_bytes_at_production, 1e-4) if total_bytes_at_production != 0 else 0.0
        
        features['total_bytes_at_production'] = total_bytes_at_production
        features['total_vectors'] = total_vectors
        features['total_parallelism'] = total_parallelism
        features['computation_efficiency'] = comp_efficiency
        features['bytes_processing_rate'] = bytes_processing_rate
        features['memory_utilization_ratio'] = mem_util_ratio
        features['sched_inner_parallelism_squared'] = total_inner_parallelism ** 2
        features['computation_efficiency_squared'] = comp_efficiency ** 2
        features['comp_efficiency_total_vectors'] = comp_efficiency * total_vectors
        features['inner_parallelism_total_parallelism'] = total_inner_parallelism * total_parallelism
        
        if total_vectors > 0:
            features['bytes_per_vector'] = total_bytes_at_production / total_vectors
        else:
            features['bytes_per_vector'] = 0
        
        if 'working_set' in scheduling_features[0] and 'bytes_at_production' in scheduling_features[0]:
            features['memory_pressure'] = scheduling_features[0]['working_set'] / scheduling_features[0]['bytes_at_production'] if scheduling_features[0]['bytes_at_production'] > 0 else 0
    
    if len(nodes_features) > 0:
        op_types = sum(1 for k in op_counts.keys())
        features['avg_ops_per_node'] = sum(op_counts.values()) / len(nodes_features)
        features['op_diversity'] = op_types / len(nodes_features) if len(nodes_features) > 0 else 0
    
    return features

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

def clean_and_transform_features(train_features, test_features):
    all_features_df = pd.DataFrame(train_features + test_features)
    
    print(f"Initial features: {list(all_features_df.columns)}")
    print(f"Initial feature count: {len(all_features_df.columns) - 1}")  # Exclude 'execution_time'
    
    all_features_df = all_features_df.fillna(0)
    
    low_importance_features = [
        'op_cast', 'op_eq', 'op_ne', 'op_or', 'op_and', 'op_le', 'op_lt', 'op_not',
        'sched_num_scalars', 'sched_bytes_at_realization', 'sched_outer_parallelism',
        'sched_num_realizations', 'sched_num_productions', 'sched_bytes_at_root'
    ]
    dropped_low_importance = [col for col in low_importance_features if col in all_features_df.columns]
    all_features_df = all_features_df.drop(columns=dropped_low_importance)
    print(f"Dropped low-importance features: {dropped_low_importance}")
    print(f"Features after dropping low-importance: {list(all_features_df.columns)}")
    
    constant_columns = [col for col in all_features_df.columns 
                       if col != 'execution_time' and all_features_df[col].nunique() == 1]
    all_features_df = all_features_df.drop(columns=constant_columns)
    print(f"Dropped constant columns: {constant_columns}")
    print(f"Features after dropping constant columns: {list(all_features_df.columns)}")
    
    corr_matrix = all_features_df.drop(['execution_time'], axis=1).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    all_features_df = all_features_df.drop(columns=to_drop)
    print(f"Dropped highly correlated features: {to_drop}")
    print(f"Features after dropping correlated features: {list(all_features_df.columns)}")
    
    skewed_features = ['computation_efficiency', 'bytes_processing_rate', 'total_parallelism', 'total_vectors', 'bytes_per_vector']
    for feature in skewed_features:
        if feature in all_features_df.columns:
            all_features_df[f'log_{feature}'] = np.log1p(all_features_df[feature])
            all_features_df = all_features_df.drop(columns=[feature])
    print(f"Features after log transformations: {list(all_features_df.columns)}")
    
    if 'execution_time' in all_features_df.columns:
        all_features_df['execution_time_log'] = np.log1p(all_features_df['execution_time'])
    
    numeric_cols = all_features_df.select_dtypes(include=['number']).columns
    all_features_df = all_features_df[numeric_cols]
    
    print(f"Final features: {list(all_features_df.columns)}")
    print(f"Final feature count (excluding execution_time and execution_time_log): {len(all_features_df.columns) - 2}")
    
    train_size = len(train_features)
    train_df = all_features_df.iloc[:train_size]
    test_df = all_features_df.iloc[train_size:]
    
    return train_df, test_df

def prepare_data_for_model(train_features, test_features):
    train_df, test_df = clean_and_transform_features(train_features, test_features)
    
    if 'execution_time_log' in train_df.columns:
        y_train = train_df['execution_time_log'].values.reshape(-1, 1)
        y_test = test_df['execution_time_log'].values.reshape(-1, 1)
        train_df = train_df.drop(['execution_time', 'execution_time_log'], axis=1)
        test_df = test_df.drop(['execution_time', 'execution_time_log'], axis=1)
        is_log_transformed = True
    else:
        y_train = train_df['execution_time'].values.reshape(-1, 1)
        y_test = test_df['execution_time'].values.reshape(-1, 1)
        train_df = train_df.drop('execution_time', axis=1)
        test_df = test_df.drop('execution_time', axis=1)
        is_log_transformed = False
    
    print("\nDebugging target values in prepare_data_for_model:")
    print(f"First 5 y_train raw: {y_train[:5].flatten()}")
    print(f"First 5 y_test raw: {y_test[:5].flatten()}")
    
    scaler_X = StandardScaler()
    scaler_y = RobustScaler()
    
    X_train_scaled = scaler_X.fit_transform(train_df)
    X_test_scaled = scaler_X.transform(test_df)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    print(f"First 5 y_train scaled: {y_train_scaled[:5].flatten()}")
    print(f"First 5 y_test scaled: {y_test_scaled[:5].flatten()}")
    
    X_train_aug = []
    y_train_aug = []
    inner_parallelism_idx = train_df.columns.get_loc('inner_parallelism_total_parallelism') if 'inner_parallelism_total_parallelism' in train_df.columns else -1
    comp_efficiency_idx = train_df.columns.get_loc('log_computation_efficiency') if 'log_computation_efficiency' in train_df.columns else -1
    
    for i in range(len(X_train_scaled)):
        X_train_aug.append(X_train_scaled[i])
        y_train_aug.append(y_train_scaled[i])
        
        is_significant = False
        if inner_parallelism_idx != -1 and X_train_scaled[i, inner_parallelism_idx] > np.percentile(X_train_scaled[:, inner_parallelism_idx], 75):
            is_significant = True
        if comp_efficiency_idx != -1 and X_train_scaled[i, comp_efficiency_idx] > np.percentile(X_train_scaled[:, comp_efficiency_idx], 75):
            is_significant = True
        
        augment_count = 3 if is_significant else 1
        for _ in range(augment_count):
            noise_x = np.random.normal(0, 0.05, X_train_scaled[i].shape)
            noise_y = np.random.normal(0, 0.05, y_train_scaled[i].shape)
            X_train_aug.append(X_train_scaled[i] + noise_x)
            y_train_aug.append(y_train_scaled[i] + noise_y)
    
    X_train_scaled = np.array(X_train_aug)
    y_train_scaled = np.array(y_train_aug)
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    print(f"Input feature dimension: {X_train_scaled.shape[1]}")
    
    return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
            scaler_y, X_train_scaled.shape[1], is_log_transformed, train_df.columns)

class PerfectLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1, dropout_rate=0.3):
        super(PerfectLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, output_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        
        x = self.fc1(lstm_out)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

class CustomMAPELoss(nn.Module):
    def __init__(self, epsilon=1e-2):
        super(CustomMAPELoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, outputs, targets, inputs, feature_indices, feature_importances):
        base_mape = torch.abs((targets - outputs) / (targets + self.epsilon))
        
        weights = torch.ones_like(targets)
        for feature, idx in feature_indices.items():
            if idx != -1 and feature in feature_importances:
                feature_vals = inputs[:, -1, idx]
                importance = feature_importances[feature]
                weights = torch.where(
                    feature_vals > 1.0,
                    weights * (1.0 + importance * 2.0),
                    weights
                )
        
        weighted_mape = (base_mape * weights).mean() * 100
        return weighted_mape

def save_checkpoint(model, optimizer, scheduler, epoch, train_losses, val_losses, best_val_loss, epochs_no_improve, checkpoint_path='checkpoint_lstm.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'epochs_no_improve': epochs_no_improve
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path='checkpoint_lstm.pth'):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Check architecture compatibility
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(checkpoint['model_state_dict'].keys())
        if model_keys != checkpoint_keys:
            print(f"Architecture mismatch! Expected keys: {model_keys}, but found: {checkpoint_keys}")
            print("Starting training from scratch due to architecture incompatibility.")
            # Optionally, delete the old checkpoint to avoid future issues
            os.remove(checkpoint_path)
            print(f"Deleted incompatible checkpoint at {checkpoint_path}")
            return 0, [], [], float('inf'), 0
        
        # Check input size compatibility
        checkpoint_input_size = None
        for key, param in checkpoint['model_state_dict'].items():
            if key == 'lstm.weight_ih_l0':
                checkpoint_input_size = param.shape[1]
                break
        current_input_size = model.lstm.weight_ih_l0.shape[1]
        if checkpoint_input_size != current_input_size:
            print(f"Input size mismatch! Checkpoint expects {checkpoint_input_size} features, but model expects {current_input_size} features.")
            print("Starting training from scratch due to input size incompatibility.")
            os.remove(checkpoint_path)
            print(f"Deleted incompatible checkpoint at {checkpoint_path}")
            return 0, [], [], float('inf'), 0
        
        # If all checks pass, load the checkpoint
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
            best_val_loss = checkpoint['best_val_loss']
            epochs_no_improve = checkpoint['epochs_no_improve']
            print(f"Loaded checkpoint from {checkpoint_path}, resuming from epoch {start_epoch}")
            return start_epoch, train_losses, val_losses, best_val_loss, epochs_no_improve
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}. Starting training from scratch.")
            os.remove(checkpoint_path)
            print(f"Deleted incompatible checkpoint at {checkpoint_path}")
            return 0, [], [], float('inf'), 0
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
        return 0, [], [], float('inf'), 0

def train_model(model, train_loader, test_loader, criterion, optimizer, feature_indices, feature_importances, num_epochs=500, patience=30, checkpoint_path='checkpoint_lstm.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    start_epoch, train_losses, val_losses, best_val_loss, epochs_no_improve = load_checkpoint(
        model, optimizer, scheduler, checkpoint_path
    )
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets, inputs, feature_indices, feature_importances)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets, inputs, feature_indices, feature_importances)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        save_checkpoint(
            model, optimizer, scheduler, epoch, train_losses, val_losses, 
            best_val_loss, epochs_no_improve, checkpoint_path
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_lstm_model.pth')
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    if os.path.exists('best_lstm_model.pth'):
        model.load_state_dict(torch.load('best_lstm_model.pth'))
        print("Loaded best model state from 'best_lstm_model.pth'")
    
    return train_losses, val_losses

def evaluate_model(model, X_test, y_test, y_scaler, file_names_test, is_log_transformed=False, original_execution_times=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    X_test = X_test.to(device)
    with torch.no_grad():
        y_pred_scaled = model(X_test)
    
    y_pred_scaled = y_pred_scaled.cpu().numpy()
    y_test = y_test.cpu().numpy()
    
    y_test_transformed = y_scaler.inverse_transform(y_test)
    y_pred_transformed = y_scaler.inverse_transform(y_pred_scaled)
    
    if is_log_transformed:
        y_test_actual = np.expm1(y_test_transformed)
        y_pred_actual = np.expm1(y_pred_transformed)
    else:
        y_test_actual = y_test_transformed
        y_pred_actual = y_pred_transformed
    
    y_pred_actual = np.maximum(y_pred_actual, 1e-2)
    
    results_by_subfolder = {}
    for i, file_path in enumerate(file_names_test):
        subfolder = file_path.split('/')[0]
        if subfolder not in results_by_subfolder:
            results_by_subfolder[subfolder] = []
        
        actual_val = y_test_actual[i][0]
        pred_val = y_pred_actual[i][0]
        error_percentage = abs(actual_val - pred_val) / actual_val * 100 if actual_val > 0 else 0
        error_percentage = min(error_percentage, 1000.0)
        
        results_by_subfolder[subfolder].append({
            'file': file_path,
            'actual': actual_val,
            'predicted': pred_val,
            'error_percentage': error_percentage
        })
    
    for subfolder, results in results_by_subfolder.items():
        print(f"\nResults for {subfolder}:")
        for result in results:
            print(f"File: {result['file']}")
            print(f"  Actual execution time: {result['actual']} ms")
            print(f"  Predicted execution time: {result['predicted']} ms")
            print(f"  Error percentage: {result['error_percentage']:.2f}%")
    
    mse = np.mean((y_test_actual - y_pred_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_actual - y_pred_actual))
    mask = y_test_actual > 1.0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_test_actual[mask] - y_pred_actual[mask]) / y_test_actual[mask])) * 100
    else:
        mape = 0.0
    
    print("\nOverall Model Performance:")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape:.2f}%")
    
    return y_test_actual, y_pred_actual

def main(main_dir):
    print(f"Processing main directory: {main_dir}")
    train_features, test_features, test_file_names = process_main_directory(main_dir)
    
    print(f"Total training samples: {len(train_features)} (randomly selected)")
    print(f"Total test samples: {len(test_features)} (50 randomly selected)")
    
    if len(train_features) == 0 or len(test_features) == 0:
        print("Error: No valid training or test data found")
        return None
    
    original_execution_times = {}
    for feature, fname in zip(test_features, test_file_names):
        original_execution_times[fname] = feature['execution_time']
    
    X_train, y_train, X_test, y_test, y_scaler, input_size, is_log_transformed, feature_columns = prepare_data_for_model(train_features, test_features)
    
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32)
    
    model = PerfectLSTMModel(
        input_size=input_size,
        hidden_size=128,
        output_size=1,
        dropout_rate=0.3
    )
    
    criterion = CustomMAPELoss(epsilon=1e-2)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    feature_importances = {
        'computation_efficiency': 0.6064,
        'inner_parallelism_total_parallelism': 0.2135,
        'total_parallelism': 0.0038,
        'total_vectors': 0.0138,
        'scheduling_count': 0.0454,
        'total_bytes_at_production': 0.0357,
        'bytes_processing_rate': 0.0064
    }
    
    feature_indices = {}
    for feature in feature_importances.keys():
        log_feature = f'log_{feature}' if feature in ['computation_efficiency', 'bytes_processing_rate', 'total_parallelism', 'total_vectors'] else feature
        if log_feature in feature_columns:
            feature_indices[feature] = feature_columns.get_loc(log_feature)
        else:
            feature_indices[feature] = feature_columns.get_loc(feature) if feature in feature_columns else -1
    
    print("Building and training Perfect LSTM model...")
    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        test_loader, 
        criterion, 
        optimizer, 
        feature_indices,
        feature_importances,
        num_epochs=500,
        patience=30,
        checkpoint_path='checkpoint_lstm.pth'
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_perfect_model.png')
    plt.close()
    print("Training plot saved as 'loss_perfect_model.png'")
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(
        model, X_test, y_test, y_scaler, test_file_names, 
        is_log_transformed, original_execution_times
    )
    
    print("\nSaving the trained model as 'lstm_model.pt'...")
    model.eval()
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")
    
    try:
        sample_input = torch.randn(1, 1, input_size).to(device)
        traced_model = torch.jit.trace(model, sample_input)
        traced_model.save("lstm_model.pt")
        print("Model successfully saved as 'lstm_model.pt'")
    except Exception as e:
        print(f"Error saving the model: {str(e)}")
    
    return model, y_scaler, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "synthetic_data"
    random.seed(42)
    result = main(main_dir)
    if result is not None:
        model, y_scaler, y_test_actual, y_pred_actual = result
