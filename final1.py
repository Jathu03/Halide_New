import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import random
import matplotlib.pyplot as plt
import pickle
import time
import psutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

set_seed(42)

def get_execution_time(file_path, max_retries=2):
    for attempt in range(max_retries):
        try:
            with open(file_path, 'rb') as f:
                raw_content = f.read()
                content = raw_content.decode('utf-8', errors='replace').replace('\0', '')
                data = json.loads(content)
            
            if 'programming_details' not in data:
                logging.error(f"'programming_details' key not found in {file_path}")
                return None
            
            schedules = data.get("scheduling_data", [])
            for item in schedules:
                if isinstance(item, dict) and item.get('name') == 'total_execution_time_ms':
                    execution_time = item.get('value')
                    if execution_time is not None:
                        logging.info(f"Extracted execution time for {file_path}: {execution_time} ms")
                        return float(execution_time)
            
            if schedules and isinstance(schedules[-1], dict) and "value" in schedules[-1]:
                execution_time = schedules[-1]["value"]
                logging.warning(f"'total_execution_time_ms' not found in {file_path}, using last schedule value: {execution_time} ms")
                return float(execution_time)
            
            logging.error(f"No valid execution time found in {file_path}")
            return None
        
        except FileNotFoundError:
            logging.error(f"File {file_path} not found")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON format in {file_path}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(0.1)
                continue
            return None
        except UnicodeDecodeError as e:
            logging.error(f"Encoding issue in {file_path}: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error in {file_path}: {str(e)}")
            return None

def extract_features_from_file(file_path, cache_dir='feature_cache'):
    cache_path = os.path.join(cache_dir, file_path.replace('/', '_').replace('.json', '.pkl'))
    if os.path.exists(cache_path) and os.path.getmtime(file_path) <= os.path.getmtime(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Error reading {file_path}: {str(e)}")
        return None
    
    execution_time = get_execution_time(file_path)
    if execution_time is None:
        return None
    
    nodes_features = []
    edges_features = []
    programming_details = data.get("programming_details", {})
    
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
    scheduling_data = data.get("scheduling_data", programming_details.get('Schedules', []))
    
    for sched in scheduling_data:
        sched_feature = {'Name': sched.get('Name', '')}
        if 'Details' in sched and 'scheduling_feature' in sched['Details']:
            sf = sched['Details']['scheduling_feature']
            for key, value in sf.items():
                sched_feature[key] = value
        scheduling_features.append(sched_feature)
    
    features = {
        'execution_time': execution_time,
        'nodes_count': len(nodes_features),
        'edges_count': len(edges_features),
        'scheduling_count': len(scheduling_features),
        'node_edge_ratio': len(nodes_features) / len(edges_features) if len(edges_features) > 0 else 0
    }
    
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
        if scheduling_features[0]:
            for metric in important_metrics:
                if metric in scheduling_features[0]:
                    features[f'sched_{metric}'] = scheduling_features[0][metric]
        
        total_bytes_at_production = sum(sf.get('bytes_at_production', 0) for sf in scheduling_features if isinstance(sf, dict))
        total_vectors = sum(sf.get('num_vectors', 0) for sf in scheduling_features if isinstance(sf, dict))
        total_parallelism = sum(sf.get('inner_parallelism', 0) * sf.get('outer_parallelism', 1) for sf in scheduling_features if isinstance(sf, dict))
        
        features['total_bytes_at_production'] = total_bytes_at_production
        features['total_vectors'] = total_vectors
        features['total_parallelism'] = total_parallelism
        features['bytes_per_vector'] = total_bytes_at_production / total_vectors if total_vectors > 0 else 0
        
        if 'working_set' in scheduling_features[0] and 'bytes_at_production' in scheduling_features[0]:
            features['memory_pressure'] = scheduling_features[0]['working_set'] / scheduling_features[0]['bytes_at_production'] if scheduling_features[0]['bytes_at_production'] > 0 else 0
    
    if len(nodes_features) > 0:
        op_types = sum(1 for k in op_counts.keys())
        features['avg_ops_per_node'] = sum(op_counts.values()) / len(nodes_features)
        features['op_diversity'] = op_types / len(nodes_features)
    
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(features, f)
    
    return features

def process_directory(directory_path, cache_dir='feature_cache'):
    all_features = []
    file_names = []
    
    json_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.json')])
    
    for filename in json_files:
        file_path = os.path.join(directory_path, filename)
        features = extract_features_from_file(file_path, cache_dir)
        if features is not None:
            all_features.append(features)
            file_names.append(filename)
    
    return all_features, file_names

def process_main_directory(main_dir, cache_dir='feature_cache', val_size=0.2):
    all_features = []
    all_file_names = []
    
    subdirs = sorted([d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))])
    if len(subdirs) < 1:
        raise ValueError(f"Expected at least 1 subdirectory in {main_dir}, found {len(subdirs)}")
    
    for subdir in subdirs:
        subdir_path = os.path.join(main_dir, subdir)
        features, file_names = process_directory(subdir_path, cache_dir)
        if not features:
            logging.warning(f"Skipping {subdir} due to no valid data")
            continue
        all_features.extend(features)
        all_file_names.extend([os.path.join(subdir, fname) for fname in file_names])
        logging.info(f"Processed subdir {subdir}: {len(features)} files")
    
    total_files = len(all_features)
    if total_files < 50:
        raise ValueError(f"Expected at least 50 files total, found {total_files}")
    
    combined = list(zip(all_features, all_file_names))
    random.shuffle(combined)
    all_features, all_file_names = zip(*combined)
    
    test_size = 50
    train_val_features = all_features[:-test_size]
    test_features = all_features[-test_size:]
    train_val_file_names = all_file_names[:-test_size]
    test_file_names = all_file_names[-test_size:]
    
    train_size = int((1 - val_size) * len(train_val_features))
    train_features = train_val_features[:train_size]
    val_features = train_val_features[train_size:]
    train_file_names = train_val_file_names[:train_size]
    val_file_names = train_val_file_names[train_size:]
    
    logging.info(f"Total files: {total_files}")
    logging.info(f"Training files: {len(train_features)}")
    logging.info(f"Validation files: {len(val_features)}")
    logging.info(f"Testing files: {len(test_features)}")
    
    return train_features, val_features, test_features, list(train_file_names), list(val_file_names), list(test_file_names)

def clean_and_transform_features(train_features, val_features, test_features):
    all_features_df = pd.DataFrame(train_features + val_features + test_features)
    all_features_df = all_features_df.fillna(0)
    
    constant_columns = [col for col in all_features_df.columns 
                       if col != 'execution_time' and all_features_df[col].nunique() == 1]
    all_features_df = all_features_df.drop(columns=constant_columns)
    logging.info(f"Dropped {len(constant_columns)} constant columns")
    
    all_features_df['execution_time_log'] = np.log1p(all_features_df['execution_time'])
    if 'total_vectors' in all_features_df.columns and all_features_df['total_vectors'].max() > 0:
        all_features_df['bytes_per_vector'] = all_features_df['total_bytes_at_production'] / (all_features_df['total_vectors'] + 1e-8)
    
    # Feature selection based on correlation with target
    numeric_cols = all_features_df.select_dtypes(include=['number']).columns
    correlations = all_features_df[numeric_cols].corr()['execution_time_log'].abs()
    selected_features = correlations[correlations > 0.1].index.tolist()
    if 'execution_time' in selected_features:
        selected_features.remove('execution_time')
    if 'execution_time_log' in selected_features:
        selected_features.remove('execution_time_log')
    all_features_df = all_features_df[selected_features + ['execution_time', 'execution_time_log']]
    logging.info(f"Selected {len(selected_features)} features based on correlation > 0.1: {selected_features}")
    
    train_size = len(train_features)
    val_size = len(val_features)
    train_df = all_features_df.iloc[:train_size]
    val_df = all_features_df.iloc[train_size:train_size + val_size]
    test_df = all_features_df.iloc[train_size + val_size:]
    
    return train_df, val_df, test_df

def prepare_data_for_model(train_features, val_features, test_features):
    train_df, val_df, test_df = clean_and_transform_features(train_features, val_features, test_features)
    
    y_train = train_df['execution_time_log'].values.reshape(-1, 1)
    y_val = val_df['execution_time_log'].values.reshape(-1, 1)
    y_test = test_df['execution_time_log'].values.reshape(-1, 1)
    train_df = train_df.drop(['execution_time', 'execution_time_log'], axis=1)
    val_df = val_df.drop(['execution_time', 'execution_time_log'], axis=1)
    test_df = test_df.drop(['execution_time', 'execution_time_log'], axis=1)
    
    logging.info("\nDebugging target values in prepare_data_for_model:")
    logging.info(f"First 5 y_train raw: {y_train[:5].flatten()}")
    logging.info(f"First 5 y_val raw: {y_val[:5].flatten()}")
    logging.info(f"First 5 y_test raw: {y_test[:5].flatten()}")
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(train_df)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_val_scaled = scaler_X.transform(val_df)
    y_val_scaled = scaler_y.transform(y_val)
    X_test_scaled = scaler_X.transform(test_df)
    y_test_scaled = scaler_y.transform(y_test)
    
    logging.info(f"First 5 y_train scaled: {y_train_scaled[:5].flatten()}")
    logging.info(f"First 5 y_val scaled: {y_val_scaled[:5].flatten()}")
    logging.info(f"First 5 y_test scaled: {y_test_scaled[:5].flatten()}")
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_val_tensor = torch.FloatTensor(X_val_scaled).unsqueeze(1)
    y_val_tensor = torch.FloatTensor(y_val_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    logging.info(f"Input feature dimension: {X_train_scaled.shape[1]}")
    
    return (X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor, 
            scaler_X, scaler_y, X_train_scaled.shape[1], True)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
    
    def forward(self, x, device):
        batch_size = x.shape[0]
        
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        self.scale = self.scale.to(device)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        attention = torch.softmax(energy, dim=-1)
        out = torch.matmul(attention, V).permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hidden_size)
        out = self.fc_out(out)
        return out

class AdvancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], output_size=1, dropout_rate=0.5, num_heads=4):
        super(AdvancedLSTMModel, self).__init__()
        
        self.hidden_sizes = hidden_sizes
        self.lstm_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        
        for i, hidden_size in enumerate(hidden_sizes):
            in_size = input_size if i == 0 else hidden_sizes[i-1] * 2  # *2 for bidirectional
            self.lstm_layers.append(nn.LSTM(in_size, hidden_size, batch_first=True, bidirectional=True))
            self.ln_layers.append(nn.LayerNorm(hidden_size * 2))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            self.attention_layers.append(MultiHeadAttention(hidden_size * 2, num_heads))
        
        self.fc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        fc_sizes = [hidden_sizes[-1] * 2, 128, 64, 32, 16]
        for i in range(len(fc_sizes) - 1):
            self.fc_layers.append(nn.Linear(fc_sizes[i], fc_sizes[i+1]))
            self.bn_layers.append(nn.BatchNorm1d(fc_sizes[i+1]))
        
        self.output_layer = nn.Linear(fc_sizes[-1], output_size)
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        device = x.device
        lstm_out = x
        
        for i, (lstm, ln, dropout, attention) in enumerate(zip(self.lstm_layers, self.ln_layers, self.dropout_layers, self.attention_layers)):
            lstm_out, _ = lstm(lstm_out)
            lstm_out = ln(lstm_out)
            lstm_out = attention(lstm_out, device)
            if i < len(self.lstm_layers) - 1:
                lstm_out = dropout(lstm_out)
        
        attn_out = lstm_out[:, -1, :]
        
        fc_out = attn_out
        for fc, bn in zip(self.fc_layers, self.bn_layers):
            fc_out = fc(fc_out)
            fc_out = bn(fc_out)
            fc_out = self.leaky_relu(fc_out)
        
        output = self.output_layer(fc_out)
        return output

def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=12):
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    try:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=4, pin_memory=True
        )
    except RuntimeError as e:
        logging.warning(f"DataLoader failed with batch_size={batch_size}, trying batch_size=8: {str(e)}")
        batch_size = 8
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=4, pin_memory=True
        )
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=350, patience=40):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    model.to(device)
    
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-4,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1000
    )
    
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    val_maes = []
    val_mapes = []
    train_mapes = []
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        train_mape = 0.0
        train_count = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
            scheduler.step()
            running_loss += loss.item() * inputs.size(0)
            train_mape += torch.abs((outputs - targets) / (targets.abs() + 1e-8)).sum().item()
            train_count += inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_mape = (train_mape / len(train_loader.dataset)) * 100
        train_losses.append(train_loss)
        train_mapes.append(train_mape)
        
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_mape = 0.0
        val_count = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                val_mae += torch.abs(outputs - targets).sum().item()
                val_mape += torch.abs((outputs - targets) / (targets.abs() + 1e-8)).sum().item()
                val_count += inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        val_mape = (val_mape / len(val_loader.dataset)) * 100
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        val_mapes.append(val_mape)
        
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        mem_usage = psutil.Process().memory_info().rss / (1024 ** 2)  # MB
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train MAPE: {train_mape:.2f}%, Val MAE: {val_mae:.4f}, Val MAPE: {val_mape:.2f}%, Grad Norm: {grad_norm:.4f}, Memory: {mem_usage:.2f} MB, Time: {time.time() - epoch_start_time:.2f}s')
        
        # Save checkpoint
        if epoch % 50 == 0:
            checkpoint_path = f'checkpoint_epoch_{epoch}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Saved checkpoint: {checkpoint_path}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            logging.info(f'Early stopping after {epoch+1} epochs')
            model.load_state_dict(best_model_state)
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses, val_maes, val_mapes, train_mapes

def evaluate_model(model, X_test, y_test, scaler_y, file_names_test, is_log_transformed=True, original_execution_times=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    X_test = X_test.to(device)
    with torch.no_grad():
        y_pred_scaled = model(X_test)
    
    y_pred_scaled = y_pred_scaled.cpu().numpy()
    y_test = y_test.cpu().numpy()
    
    y_test_transformed = scaler_y.inverse_transform(y_test)
    y_pred_transformed = scaler_y.inverse_transform(y_pred_scaled)
    
    logging.info("\nDebugging transformed values before inverse log:")
    for i in range(min(5, len(y_test_transformed))):
        logging.info(f"Sample {i}: y_test_transformed={y_test_transformed[i][0]:.4f}, y_pred_transformed={y_pred_transformed[i][0]:.4f}")
    
    y_test_actual = np.expm1(y_test_transformed)
    y_pred_actual = np.expm1(y_pred_transformed)
    
    logging.info("\nDebugging final values after all transformations:")
    for i in range(min(5, len(y_test_actual))):
        logging.info(f"Sample {i}: y_test_actual={y_test_actual[i][0]:.4f}, y_pred_actual={y_pred_actual[i][0]:.4f}")
        if original_execution_times:
            logging.info(f"  Original execution time from JSON: {original_execution_times[file_names_test[i]]:.4f}")
    
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
        logging.info(f"\nResults for {subfolder}:")
        for result in results:
            logging.info(f"File: {result['file']}")
            logging.info(f"  Actual execution time: {result['actual']:.4f} ms")
            logging.info(f"  Predicted execution time: {result['predicted']:.4f} ms")
            logging.info(f"  Error percentage: {result['error_percentage']:.2f}%")
    
    mse = np.mean((y_test_actual - y_pred_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_actual - y_pred_actual))
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / (y_test_actual + 1e-8))) * 100
    
    logging.info("\nOverall Model Performance:")
    logging.info(f"MSE: {mse:.4f}")
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"MAE: {mae:.4f}")
    logging.info(f"MAPE: {mape:.2f}%")
    
    # Plot error histogram
    errors = np.abs(y_test_actual - y_pred_actual).flatten()
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, density=True, alpha=0.75)
    plt.title('Histogram of Absolute Prediction Errors')
    plt.xlabel('Absolute Error (ms)')
    plt.ylabel('Density')
    plt.grid(True)
    plt.savefig('error_histogram_advanced_lstm_fixed.png')
    plt.close()
    logging.info("Error histogram saved as 'error_histogram_advanced_lstm_fixed.png'")
    
    return y_test_actual, y_pred_actual

def plot_metrics(train_losses, val_losses, val_maes, val_mapes, train_mapes):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(range(1, len(val_maes) + 1), val_maes, label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Validation MAE')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(range(1, len(train_mapes) + 1), train_mapes, label='Training MAPE')
    plt.plot(range(1, len(val_mapes) + 1), val_mapes, label='Validation MAPE')
    plt.xlabel('Epoch')
    plt.ylabel('MAPE (%)')
    plt.title('Training and Validation MAPE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('metrics_advanced_lstm_fixed.png')
    plt.close()
    logging.info("Metrics plot saved as 'metrics_advanced_lstm_fixed.png'")

def main(main_dir, cache_dir='feature_cache'):
    logging.info(f"Processing main directory: {main_dir}")
    train_features, val_features, test_features, train_file_names, val_file_names, test_file_names = process_main_directory(main_dir, cache_dir)
    
    logging.info(f"Total training samples: {len(train_features)}")
    logging.info(f"Total validation samples: {len(val_features)}")
    logging.info(f"Total test samples: {len(test_features)}")
    
    if len(train_features) == 0 or len(test_features) == 0:
        logging.error("No valid training or test data found")
        return None
    
    original_execution_times = {fname: f['execution_time'] for f, fname in zip(test_features, test_file_names)}
    
    (X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y, input_size, is_log_transformed) = prepare_data_for_model(
        train_features, val_features, test_features
    )
    
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=12
    )
    
    model = AdvancedLSTMModel(
        input_size=input_size,
        hidden_sizes=[512, 256, 128],
        output_size=1,
        dropout_rate=0.5,
        num_heads=4
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    
    logging.info("Building and training Advanced LSTM model...")
    train_losses, val_losses, val_maes, val_mapes, train_mapes = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=350, patience=40
    )
    
    plot_metrics(train_losses, val_losses, val_maes, val_mapes, train_mapes)
    
    logging.info("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(
        model, X_test, y_test, scaler_y, test_file_names, is_log_transformed, original_execution_times
    )
    
    logging.info("\nSaving the trained model as 'advanced_lstm_model_fixed.pt'...")
    model.eval()
    device = next(model.parameters()).device
    logging.info(f"Model is on device: {device}")
    
    try:
        sample_input = torch.randn(1, 1, input_size).to(device)
        traced_model = torch.jit.trace(model, sample_input)
        traced_model.save("advanced_lstm_model_fixed.pt")
        logging.info("Model successfully saved as 'advanced_lstm_model_fixed.pt'")
    except Exception as e:
        logging.error(f"Error saving the model: {str(e)}")
    
    # Save scalers
    with open('scaler_X_fixed.pkl', 'wb') as f:
        pickle.dump(scaler_X, f)
    with open('scaler_y_fixed.pkl', 'wb') as f:
        pickle.dump(scaler_y, f)
    
    return model, scaler_y, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "synthetic_data"
    result = main(main_dir)
    if result is not None:
        model, scaler_y, y_test_actual, y_pred_actual = result
