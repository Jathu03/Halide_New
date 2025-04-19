import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import random
import matplotlib.pyplot as plt
import re

def get_execution_time(file_path):
    try:
        with open(file_path, 'rb') as f:
            raw_content = f.read()
            content = raw_content.decode('utf-8', errors='replace').replace('\0', '')
            data = json.loads(content)
        
        schedules = data.get("scheduling_data", [])
        for item in schedules:
            if isinstance(item, dict) and item.get('name') == 'total_execution_time_ms':
                execution_time = item.get('value')
                if execution_time is not None:
                    print(f"Extracted execution time for {file_path}: {execution_time} ms")
                    return float(execution_time)
        
        if 'programming_details' in data:
            prog_details = data['programming_details']
            if 'Schedules' in prog_details:
                for item in prog_details['Schedules']:
                    if isinstance(item, dict) and item.get('Name') == 'total_execution_time_ms':
                        execution_time = item.get('Value')
                        if execution_time is not None:
                            print(f"Extracted execution time from programming_details for {file_path}: {execution_time} ms")
                            return float(execution_time)
        
        if schedules and isinstance(schedules[-1], dict) and "value" in schedules[-1]:
            execution_time = schedules[-1]["value"]
            print(f"Warning: 'total_execution_time_ms' not found, using last schedule value: {execution_time} ms")
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

def extract_op_histogram(op_hist_list):
    op_counts = {}
    for op_line in op_hist_list:
        if isinstance(op_line, str):
            parts = re.split(r'[:=]', op_line.strip(), 1)
            if len(parts) == 2:
                op_name = parts[0].strip().lower()
                try:
                    op_count = int(parts[1].strip())
                    op_counts[f'op_{op_name}'] = op_count
                except ValueError:
                    match = re.search(r'\d+', parts[1])
                    if match:
                        op_counts[f'op_{op_name}'] = int(match.group())
    return op_counts

def extract_features_from_file(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        execution_time = get_execution_time(file_path)
        if execution_time is None:
            print(f"Warning: No execution time found in {file_path}")
            return None
        
        features = {'execution_time': execution_time}
        nodes_features = []
        edges_features = []
        programming_details = data.get("programming_details", {})
        
        if 'Nodes' in programming_details:
            for node in programming_details['Nodes']:
                node_feature = {}
                if 'Details' in node and 'Op histogram' in node['Details']:
                    op_hist = node['Details']['Op histogram']
                    if isinstance(op_hist, list):
                        node_feature.update(extract_op_histogram(op_hist))
                    elif isinstance(op_hist, str):
                        node_feature.update(extract_op_histogram([op_hist]))
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
        scheduling_data = data.get("scheduling_data", [])
        if not scheduling_data and 'Schedules' in programming_details:
            scheduling_data = programming_details['Schedules']
        
        if scheduling_data:
            for sched in scheduling_data:
                if not isinstance(sched, dict):
                    continue
                sched_feature = {}
                if 'Details' in sched and 'scheduling_feature' in sched['Details']:
                    sf = sched['Details']['scheduling_feature']
                    if isinstance(sf, dict):
                        for key, value in sf.items():
                            if isinstance(value, (int, float)):
                                sched_feature[key] = value
                scheduling_features.append(sched_feature)
        
        features['nodes_count'] = len(nodes_features)
        features['edges_count'] = len(edges_features)
        features['scheduling_count'] = len(scheduling_features)
        features['node_edge_ratio'] = len(nodes_features) / len(edges_features) if len(edges_features) > 0 else 0
        
        features['nodes_edges_interaction'] = features['nodes_count'] * features['edges_count']
        
        op_counts = {}
        for node in nodes_features:
            for key, value in node.items():
                if key.startswith('op_'):
                    op_counts[key] = op_counts.get(key, 0) + value
        features.update(op_counts)
        
        if edges_features:
            node_degrees = {}
            for edge in edges_features:
                from_node = edge['From']
                to_node = edge['To']
                node_degrees[from_node] = node_degrees.get(from_node, 0) + 1
                node_degrees[to_node] = node_degrees.get(to_node, 0) + 1
            degrees = list(node_degrees.values())
            features['avg_node_degree'] = np.mean(degrees) if degrees else 0
            features['max_node_degree'] = max(degrees) if degrees else 0
        
        if scheduling_features:
            important_metrics = [
                'bytes_at_production', 'num_vectors', 'inner_parallelism', 'outer_parallelism',
                'points_computed_total', 'working_set'
            ]
            if scheduling_features[0]:
                for metric in important_metrics:
                    if metric in scheduling_features[0]:
                        features[f'sched_{metric}'] = scheduling_features[0][metric]
            
            total_bytes_at_production = sum(sf.get('bytes_at_production', 0) for sf in scheduling_features)
            total_vectors = sum(sf.get('num_vectors', 0) for sf in scheduling_features)
            features['total_bytes_at_production'] = total_bytes_at_production
            features['total_vectors'] = total_vectors
            features['bytes_per_vector'] = total_bytes_at_production / total_vectors if total_vectors > 0 else 0
        
        if len(nodes_features) > 0:
            features['avg_ops_per_node'] = sum(op_counts.values()) / len(nodes_features)
        
        return features
    
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

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
    all_features_df = all_features_df.fillna(0)
    
    constant_columns = [col for col in all_features_df.columns 
                       if col != 'execution_time' and all_features_df[col].nunique() == 1]
    all_features_df = all_features_df.drop(columns=constant_columns)
    print(f"Dropped {len(constant_columns)} constant columns")
    
    all_features_df['execution_time_log'] = np.log1p(all_features_df['execution_time'])
    
    corr_matrix = all_features_df.drop(['execution_time', 'execution_time_log'], axis=1).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]
    all_features_df = all_features_df.drop(columns=to_drop)
    print(f"Dropped {len(to_drop)} highly correlated features")
    
    numeric_cols = all_features_df.select_dtypes(include=['number']).columns
    all_features_df = all_features_df[numeric_cols]
    
    train_size = len(train_features)
    train_df = all_features_df.iloc[:train_size]
    test_df = all_features_df.iloc[train_size:]
    
    return train_df, test_df

def prepare_data_for_model(train_features, test_features):
    train_df, test_df = clean_and_transform_features(train_features, test_features)
    
    y_train = train_df['execution_time_log'].values.reshape(-1, 1)
    y_test = test_df['execution_time_log'].values.reshape(-1, 1)
    train_df = train_df.drop(['execution_time', 'execution_time_log'], axis=1)
    test_df = test_df.drop(['execution_time', 'execution_time_log'], axis=1)
    
    print("\nDebugging target values:")
    print(f"First 5 y_train raw: {y_train[:5].flatten()}")
    print(f"First 5 y_test raw: {y_test[:5].flatten()}")
    
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    
    X_train_scaled = scaler_X.fit_transform(train_df)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(test_df)
    y_test_scaled = scaler_y.transform(y_test)
    
    print(f"First 5 y_train scaled: {y_train_scaled[:5].flatten()}")
    print(f"First 5 y_test scaled: {y_test_scaled[:5].flatten()}")
    
    if train_df.shape[1] > 30:
        selector = SelectKBest(f_regression, k=30)
        X_train_scaled = selector.fit_transform(X_train_scaled, y_train.flatten())
        X_test_scaled = selector.transform(X_test_scaled)
        selected_mask = selector.get_support()
        selected_features = [train_df.columns[i] for i in range(len(train_df.columns)) if selected_mask[i]]
        print(f"Selected {len(selected_features)} features: {selected_features[:10]}...")
    
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    print(f"Input feature dimension: {X_train_scaled.shape[1]}")
    
    return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
            scaler_X, scaler_y, train_df.columns.tolist(), True)

class SimplifiedHybridModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1, dropout_rate=0.3):
        super(SimplifiedHybridModel, self).__init__()
        self.hidden_size = hidden_size
        
        # Simple MLP with batch normalization
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        # Input shape: [batch_size, feature_dim]
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=16):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

class CustomLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(CustomLoss, self).__init__()
        self.huber = nn.HuberLoss(delta=delta)
    
    def forward(self, outputs, targets):
        return self.huber(outputs, targets)

class LinearWarmup:
    def __init__(self, optimizer, warmup_epochs, start_lr, max_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.start_lr = start_lr
        self.max_lr = max_lr
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.start_lr + (self.max_lr - self.start_lr) * self.current_epoch / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            print(f"Warmup epoch {self.current_epoch}: Learning rate set to {lr:.6f}")

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=200, patience=30):
    device = torch.device('cpu')
    model.to(device)
    
    warmup = LinearWarmup(optimizer, warmup_epochs=10, start_lr=1e-5, max_lr=0.001)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=2, eta_min=1e-5)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
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
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
ワイ        if epoch < 10:
            warmup.step()
        else:
            scheduler.step(epoch)
            print(f"Epoch {epoch+1}: Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses

def evaluate_model(model, X_test, y_test, y_scaler, file_names_test, is_log_transformed, original_execution_times):
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        y_pred_scaled = model(X_test.to(device)).cpu().numpy()
    
    y_test = y_test.cpu().numpy()
    # Inverse transform the scaled values
    y_test_transformed = y_scaler.inverse_transform(y_test)
    y_pred_transformed = y_scaler.inverse_transform(y_pred_scaled)
    
    # Inverse log transform
    y_test_actual = np.expm1(y_test_transformed) if is_log_transformed else y_test_transformed
    y_pred_actual = np.expm1(y_pred_transformed) if is_log_transformed else y_pred_transformed
    
    # Clip predictions to avoid negative values
    y_pred_actual = np.maximum(y_pred_actual, 0)
    
    results_by_subfolder = {}
    for i, file_path in enumerate(file_names_test):
        subfolder = file_path.split('/')[0]
        if subfolder not in results_by_subfolder:
            results_by_subfolder[subfolder] = []
        
        actual_val = y_test_actual[i][0]
        pred_val = y_pred_actual[i][0]
        error_percentage = abs(actual_val - pred_val) / actual_val * 100 if actual_val > 0 else 0
        
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
            print(f"  Actual: {result['actual']:.2f} ms")
            print(f"  Predicted: {result['predicted']:.2f} ms")
            print(f"  Error: {result['error_percentage']:.2f}%")
    
    mse = np.mean((y_test_actual - y_pred_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_actual - y_pred_actual))
    # Use a more robust MAPE calculation
    mask = y_test_actual > 1e-2  # Avoid division by very small numbers
    mape = np.mean(np.abs((y_test_actual[mask] - y_pred_actual[mask]) / y_test_actual[mask])) * 100 if mask.sum() > 0 else 0
    
    print("\nOverall Model Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    return y_test_actual, y_pred_actual

def save_scaler(scaler, feature_names, filename):
    scaler_data = {
        'feature_names': feature_names,
        'means': scaler.center_.tolist(),
        'scales': scaler.scale_.tolist()
    }
    with open(filename, 'w') as f:
        json.dump(scaler_data, f, indent=4)
    print(f"Saved scaler to {filename}")

def save_y_scaler(scaler, is_log_transformed, filename):
    scaler_data = {
        'mean': float(scaler.center_[0]),
        'scale': float(scaler.scale_[0]),
        'is_log_transformed': is_log_transformed
    }
    with open(filename, 'w') as f:
        json.dump(scaler_data, f, indent=4)
    print(f"Saved y scaler to {filename}")

def main(main_dir):
    print(f"Processing main directory: {main_dir}")
    train_features, test_features, test_file_names = process_main_directory(main_dir)
    
    original_execution_times = {fname: f['execution_time'] for f, fname in zip(test_features, test_file_names)}
    
    (X_train, y_train, X_test, y_test, scaler_X, scaler_y, feature_names, is_log_transformed) = prepare_data_for_model(train_features, test_features)
    
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32)
    
    model = SimplifiedHybridModel(
        input_size=len(feature_names),
        hidden_size=64,
        output_size=1,
        dropout_rate=0.3
    )
    
    criterion = CustomLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    print("Training model...")
    train_losses, val_losses = train_model(
        model, train_loader, test_loader, criterion, optimizer, num_epochs=200, patience=30
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_model.png')
    plt.close()
    print("Saved loss plot as 'loss_model.png'")
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(
        model, X_test, y_test, scaler_y, test_file_names, 
        is_log_transformed, original_execution_times
    )
    
    model.eval()
    model.to('cpu')
    try:
        sample_input = torch.randn(1, len(feature_names)).to('cpu')
        traced_model = torch.jit.trace(model, sample_input)
        traced_model.save("lstm_model.pt")
        print("Saved best model as 'lstm_model.pt'")
    except Exception as e:
        print(f"Error saving best model: {str(e)}")
    
    save_scaler(scaler_X, feature_names, "scaler_X.json")
    save_y_scaler(scaler_y, is_log_transformed, "scaler_y.json")
    
    return model, scaler_y, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "synthetic_data"
    random.seed(42)
    result = main(main_dir)
    if result is not None:
        model, y_scaler, y_test_actual, y_pred_actual = result
