import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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
        
        features['total_bytes_at_production'] = total_bytes_at_production
        features['total_vectors'] = total_vectors
        features['total_parallelism'] = total_parallelism
        
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
    
    all_features_df = all_features_df.fillna(0)
    
    constant_columns = [col for col in all_features_df.columns 
                       if col != 'execution_time' and all_features_df[col].nunique() == 1]
    all_features_df = all_features_df.drop(columns=constant_columns)
    print(f"Dropped {len(constant_columns)} constant columns")
    
    if 'execution_time' in all_features_df.columns:
        all_features_df['execution_time_log'] = np.log1p(all_features_df['execution_time'])
    
    if 'total_vectors' in all_features_df.columns and all_features_df['total_vectors'].max() > 0:
        all_features_df['bytes_per_vector'] = all_features_df['total_bytes_at_production'] / (all_features_df['total_vectors'] + 1e-8)
    
    numeric_cols = all_features_df.select_dtypes(include=['number']).columns
    all_features_df = all_features_df[numeric_cols]
    
    # Dynamic outlier clipping using IQR
    for col in numeric_cols:
        if col not in ['execution_time', 'execution_time_log']:
            Q1 = all_features_df[col].quantile(0.25)
            Q3 = all_features_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            all_features_df[col] = all_features_df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Feature selection using RandomForest
    X = all_features_df.drop(['execution_time', 'execution_time_log'], axis=1, errors='ignore')
    y = all_features_df['execution_time_log'] if 'execution_time_log' in all_features_df else all_features_df['execution_time']
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    threshold = importances.quantile(0.15)  # Drop bottom 15% of features
    selected_features = importances[importances > threshold].index.tolist()
    print(f"Selected {len(selected_features)} features out of {len(X.columns)}")
    all_features_df = all_features_df[selected_features + ['execution_time', 'execution_time_log'] if 'execution_time_log' in all_features_df else ['execution_time']]
    
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
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(train_df)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(test_df)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Add noise to training data for augmentation
    X_train_scaled += np.random.normal(0, 0.01, X_train_scaled.shape)
    
    print(f"First 5 y_train scaled: {y_train_scaled[:5].flatten()}")
    print(f"First 5 y_test scaled: {y_test_scaled[:5].flatten()}")
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    print(f"Input feature dimension: {X_train_scaled.shape[1]}")
    
    return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
            scaler_X, scaler_y, train_df.columns.tolist(), is_log_transformed)

class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=1, dropout_rate=0.3):
        super(EnhancedLSTMModel, self).__init__()
        
        self.lstm_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        self.lstm_layers.append(nn.LSTM(input_size, hidden_sizes[0], batch_first=True, bidirectional=True))
        self.ln_layers.append(nn.LayerNorm(hidden_sizes[0] * 2))  # Bidirectional doubles the hidden size
        self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        for i in range(1, len(hidden_sizes)):
            self.lstm_layers.append(nn.LSTM(hidden_sizes[i-1] * 2, hidden_sizes[i], batch_first=True, bidirectional=True))
            self.ln_layers.append(nn.LayerNorm(hidden_sizes[i] * 2))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        self.attention = nn.Linear(hidden_sizes[-1] * 2, 1)
        
        self.fc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.fc_dropout = nn.Dropout(dropout_rate)
        
        self.fc_layers.append(nn.Linear(hidden_sizes[-1] * 2, hidden_sizes[-1]))
        self.bn_layers.append(nn.BatchNorm1d(hidden_sizes[-1]))
        
        self.fc_layers.append(nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2))
        self.bn_layers.append(nn.BatchNorm1d(hidden_sizes[-1] // 2))
        
        self.output_layer = nn.Linear(hidden_sizes[-1] // 2, output_size)
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
        self.has_residual = (hidden_sizes[-1] // 2 == hidden_sizes[-1])
        if not self.has_residual:
            self.residual_adapter = nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2)
        
    def attention_net(self, lstm_output):
        attn_weights = self.attention(lstm_output).squeeze(2)
        soft_attn_weights = torch.softmax(attn_weights, 1)
        context = torch.bmm(soft_attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context
        
    def forward(self, x):
        lstm_out = x
        for i, (lstm, ln, dropout) in enumerate(zip(self.lstm_layers, self.ln_layers, self.dropout_layers)):
            lstm_out, _ = lstm(lstm_out)
            lstm_out = ln(lstm_out)
            if i < len(self.lstm_layers) - 1:
                lstm_out = dropout(lstm_out)
        
        attn_output = self.attention_net(lstm_out)
        
        fc_out = self.fc_layers[0](attn_output)
        fc_out = self.bn_layers[0](fc_out)
        fc_out = self.leaky_relu(fc_out)
        fc_out = self.fc_dropout(fc_out)
        
        residual = fc_out
        if not self.has_residual:
            residual = self.residual_adapter(residual)
        
        fc_out = self.fc_layers[1](fc_out)
        fc_out = self.bn_layers[1](fc_out)
        fc_out = self.leaky_relu(fc_out)
        fc_out = self.fc_dropout(fc_out)
        
        fc_out = fc_out + residual
        
        output = self.output_layer(fc_out)
        
        return output

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=16):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

class CustomLoss(nn.Module):
    def __init__(self, delta=0.7, mse_weight=0.3, l1_weight=1e-5):
        super(CustomLoss, self).__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.mse = nn.MSELoss()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
    
    def forward(self, outputs, targets):
        huber_loss = self.huber(outputs, targets)
        mse_loss = self.mse(outputs, targets)
        l1_loss = torch.mean(torch.abs(outputs))
        return huber_loss + self.mse_weight * mse_loss + self.l1_weight * l1_loss

class LinearWarmup:
    def __init__(self, optimizer, warmup_epochs, total_epochs, start_lr, max_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
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

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=500, patience=100, warmup_epochs=20):
    device = torch.device('cpu')  # Force CPU
    print(f"Using device: {device}")
    model.to(device)
    
    warmup = LinearWarmup(optimizer, warmup_epochs, num_epochs, start_lr=1e-5, max_lr=0.0005)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    checkpoint_counter = 1
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
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
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
        
        if epoch < warmup_epochs:
            warmup.step()
        else:
            scheduler.step(epoch)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}: CosineAnnealingWarmRestarts learning rate: {current_lr:.6f}")
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
            # Save checkpoint for ensemble
            model.eval()
            model.to('cpu')
            try:
                sample_input = torch.randn(1, 1, inputs.shape[2]).to('cpu')
                traced_model = torch.jit.trace(model, sample_input)
                checkpoint_path = f"lstm_model_{checkpoint_counter}.pt"
                traced_model.save(checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
                checkpoint_counter = checkpoint_counter % 3 + 1  # Save up to 3 checkpoints
            except Exception as e:
                print(f"Error saving checkpoint: {str(e)}")
            model.to(device)
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            model.load_state_dict(best_model_state)
            break
    
    if best_model_state is not None and epochs_no_improve > 0:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses

def evaluate_model(model, X_test, y_test, y_scaler, file_names_test, is_log_transformed=False, original_execution_times=None, num_checkpoints=3):
    device = torch.device('cpu')  # Force CPU
    model.to(device)
    model.eval()
    
    # Ensemble prediction using multiple checkpoints
    y_pred_scaled_ensemble = None
    for i in range(1, num_checkpoints + 1):
        checkpoint_path = f"lstm_model_{i}.pt"
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} not found, skipping")
            continue
        try:
            checkpoint_model = torch.jit.load(checkpoint_path, map_location=device)
            checkpoint_model.eval()
            with torch.no_grad():
                y_pred_scaled = checkpoint_model(X_test.to(device)).cpu().numpy()
            if y_pred_scaled_ensemble is None:
                y_pred_scaled_ensemble = y_pred_scaled
            else:
                y_pred_scaled_ensemble += y_pred_scaled
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {str(e)}")
    
    if y_pred_scaled_ensemble is None:
        print("No checkpoints loaded, falling back to single model")
        with torch.no_grad():
            y_pred_scaled_ensemble = model(X_test).cpu().numpy()
    else:
        y_pred_scaled_ensemble /= max(1, sum(os.path.exists(f"lstm_model_{i}.pt") for i in range(1, num_checkpoints + 1)))
    
    y_test = y_test.cpu().numpy()
    
    y_test_transformed = y_scaler.inverse_transform(y_test)
    y_pred_transformed = y_scaler.inverse_transform(y_pred_scaled_ensemble)
    
    print("\nDebugging transformed values before inverse log:")
    for i in range(min(5, len(y_test_transformed))):
        print(f"Sample {i}: y_test_transformed={y_test_transformed[i][0]}, y_pred_transformed={y_pred_transformed[i][0]}")
    
    if is_log_transformed:
        y_test_actual = np.expm1(y_test_transformed)
        y_pred_actual = np.expm1(y_pred_transformed)
    else:
        y_test_actual = y_test_transformed
        y_pred_actual = y_pred_transformed
    
    print("\nDebugging final values after all transformations:")
    for i in range(min(5, len(y_test_actual))):
        print(f"Sample {i}: y_test_actual={y_test_actual[i][0]}, y_pred_actual={y_pred_actual[i][0]}")
        if original_execution_times:
            print(f"  Original execution time from JSON: {original_execution_times[file_names_test[i]]}")
    
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
            print(f"  Actual execution time: {result['actual']} ms")
            print(f"  Predicted execution time: {result['predicted']} ms")
            print(f"  Error percentage: {result['error_percentage']:.2f}%")
    
    mse = np.mean((y_test_actual - y_pred_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_actual - y_pred_actual))
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / (y_test_actual + 1e-8))) * 100
    
    print("\nOverall Model Performance:")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape:.2f}%")
    
    return y_test_actual, y_pred_actual

def save_scaler(scaler, feature_names, filename):
    scaler_data = {
        'feature_names': feature_names,
        'means': scaler.mean_.tolist(),
        'scales': scaler.scale_.tolist()
    }
    with open(filename, 'w') as f:
        json.dump(scaler_data, f, indent=4)
    print(f"Saved scaler to {filename}")

def save_y_scaler(scaler, is_log_transformed, filename):
    scaler_data = {
        'mean': float(scaler.mean_[0]),
        'scale': float(scaler.scale_[0]),
        'is_log_transformed': is_log_transformed
    }
    with open(filename, 'w') as f:
        json.dump(scaler_data, f, indent=4)
    print(f"Saved y scaler to {filename}")

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
    
    (X_train, y_train, X_test, y_test, scaler_X, scaler_y, feature_names, is_log_transformed) = prepare_data_for_model(train_features, test_features)
    
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=16)
    
    model = EnhancedLSTMModel(
        input_size=len(feature_names),
        hidden_sizes=[128, 64, 32],
        output_size=1,
        dropout_rate=0.3
    )
    
    criterion = CustomLoss(delta=0.7, mse_weight=0.3, l1_weight=1e-5)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    print("Building and training Enhanced LSTM model...")
    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        test_loader, 
        criterion, 
        optimizer, 
        num_epochs=500,
        patience=100,
        warmup_epochs=20
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_enhanced_model.png')
    plt.close()
    print("Training plot saved as 'loss_enhanced_model.png'")
    
    print("\nEvaluating model with ensemble prediction:")
    y_test_actual, y_pred_actual = evaluate_model(
        model, X_test, y_test, scaler_y, test_file_names, 
        is_log_transformed, original_execution_times, num_checkpoints=3
    )
    
    print("\nSaving the best model and scalers...")
    model.eval()
    model.to('cpu')
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")
    
    try:
        sample_input = torch.randn(1, 1, len(feature_names)).to(device)
        traced_model = torch.jit.trace(model, sample_input)
        traced_model.save("lstm_model.pt")
        print("Best model saved as 'lstm_model.pt'")
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
