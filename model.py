import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union

class EnhancedLSTMModel(nn.Module):
    """
    Enhanced LSTM model with attention and residual connections.
    """
    def __init__(self, input_size: int, hidden_sizes: List[int] = [512, 256, 128, 64], 
                 output_size: int = 1, dropout_rate: float = 0.3):
        super(EnhancedLSTMModel, self).__init__()
        
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()
        
        self.lstm_layers.append(nn.LSTM(input_size, hidden_sizes[0], batch_first=True, bidirectional=True))
        self.dropout_layers.append(nn.Dropout(dropout_rate))
        self.ln_layers.append(nn.LayerNorm(hidden_sizes[0] * 2))  # Bidirectional doubles size
        
        for i in range(1, len(hidden_sizes)):
            self.lstm_layers.append(nn.LSTM(hidden_sizes[i-1] * 2, hidden_sizes[i], batch_first=True, bidirectional=True))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            self.ln_layers.append(nn.LayerNorm(hidden_sizes[i] * 2))
        
        self.attention = nn.MultiheadAttention(hidden_sizes[-1] * 2, num_heads=8, dropout=dropout_rate, batch_first=True)
        
        self.fc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        self.fc_layers.append(nn.Linear(hidden_sizes[-1] * 2, hidden_sizes[-1]))
        self.bn_layers.append(nn.BatchNorm1d(hidden_sizes[-1]))
        
        self.fc_layers.append(nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2))
        self.bn_layers.append(nn.BatchNorm1d(hidden_sizes[-1] // 2))
        
        self.output_layer = nn.Linear(hidden_sizes[-1] // 2, output_size)
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        """Forward pass through the network."""
        lstm_out = x
        for i, (lstm, dropout, ln) in enumerate(zip(self.lstm_layers, self.dropout_layers, self.ln_layers)):
            lstm_out, _ = lstm(lstm_out)
            lstm_out = ln(lstm_out)
            lstm_out = dropout(lstm_out)
        
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_output = attn_output[:, -1, :]  # Take the last timestep
        
        fc_out = self.fc_layers[0](attn_output)
        fc_out = self.bn_layers[0](fc_out)
        fc_out = self.leaky_relu(fc_out)
        
        fc_out = self.fc_layers[1](fc_out)
        fc_out = self.bn_layers[1](fc_out)
        fc_out = self.leaky_relu(fc_out)
        
        output = self.output_layer(fc_out)
        
        return output

class EnhancedTransformerModel(nn.Module):
    """
    Enhanced Transformer model for execution time prediction.
    """
    def __init__(self, input_size: int, hidden_size: int = 256, num_heads: int = 8, 
                 num_layers: int = 4, dropout_rate: float = 0.3, output_size: int = 1):
        super(EnhancedTransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2)
        )
        
        self.output_layer = nn.Linear(hidden_size // 4, output_size)
    
    def forward(self, x):
        """Forward pass through the transformer network."""
        x = self.input_projection(x)
        transformer_output = self.transformer_encoder(x)
        output = transformer_output[:, -1, :]  # Take the last timestep
        output = self.fc_layers(output)
        output = self.output_layer(output)
        return output

class EnsembleModel:
    """
    Ensemble model combining LSTM and Transformer predictions.
    """
    def __init__(self, lstm_model, transformer_model, weight_lstm=0.5):
        self.lstm_model = lstm_model
        self.transformer_model = transformer_model
        self.weight_lstm = weight_lstm
        self.weight_transformer = 1.0 - weight_lstm
    
    def predict(self, X, device):
        self.lstm_model.eval()
        self.transformer_model.eval()
        with torch.no_grad():
            lstm_pred = self.lstm_model(X.to(device))
            transformer_pred = self.transformer_model(X.to(device))
            ensemble_pred = (self.weight_lstm * lstm_pred + 
                           self.weight_transformer * transformer_pred)
        return ensemble_pred

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
    
    # Advanced feature engineering
    features['nodes_edges_interaction'] = features['nodes_count'] * features['edges_count']
    features['nodes_count_log'] = np.log1p(features['nodes_count'])
    features['edges_count_log'] = np.log1p(features['edges_count'])
    
    if len(nodes_features) > 0 and len(edges_features) > 0:
        features['node_edge_ratio'] = len(nodes_features) / len(edges_features)
    else:
        features['node_edge_ratio'] = 0
    
    op_counts = {}
    op_complexity = {'add': 1, 'mul': 2, 'div': 3, 'sub': 1, 'conv': 5, 'relu': 2, 'sigmoid': 3, 'tanh': 3}
    total_complexity = 0
    for node in nodes_features:
        for key, value in node.items():
            if key.startswith('op_'):
                op_counts[key] = op_counts.get(key, 0) + value
                op_name = key.replace('op_', '')
                total_complexity += value * op_complexity.get(op_name, 1)
    features.update(op_counts)
    features['avg_op_complexity'] = total_complexity / len(nodes_features) if nodes_features else 0
    
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
                    features[f'sched_{metric}_log'] = np.log1p(scheduling_features[0][metric])
        
        total_bytes_at_production = sum(sf.get('bytes_at_production', 0) for sf in scheduling_features if isinstance(sf, dict))
        total_vectors = sum(sf.get('num_vectors', 0) for sf in scheduling_features if isinstance(sf, dict))
        total_parallelism = sum(sf.get('inner_parallelism', 0) * sf.get('outer_parallelism', 1) for sf in scheduling_features if isinstance(sf, dict))
        
        features['total_bytes_at_production'] = total_bytes_at_production
        features['total_vectors'] = total_vectors
        features['total_parallelism'] = total_parallelism
        features['total_bytes_at_production_log'] = np.log1p(total_bytes_at_production)
        
        if total_vectors > 0:
            features['bytes_per_vector'] = total_bytes_at_production / total_vectors
            features['bytes_per_vector_log'] = np.log1p(features['bytes_per_vector'])
        else:
            features['bytes_per_vector'] = 0
            features['bytes_per_vector_log'] = 0
        
        if 'working_set' in scheduling_features[0] and 'bytes_at_production' in scheduling_features[0]:
            features['memory_pressure'] = scheduling_features[0]['working_set'] / scheduling_features[0]['bytes_at_production'] if scheduling_features[0]['bytes_at_production'] > 0 else 0
    
    if len(nodes_features) > 0:
        op_types = sum(1 for k in op_counts.keys())
        features['avg_ops_per_node'] = sum(op_counts.values()) / len(nodes_features)
        features['op_diversity'] = op_types / len(nodes_features)
    
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
    
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
        print(f"Created directory: {main_dir}")
        subdirs = ["workload_A", "workload_B", "workload_C"]
        for subdir in subdirs:
            subdir_path = os.path.join(main_dir, subdir)
            if not os.path.exists(subdir_path):
                os.makedirs(subdir_path)
                print(f"Created subdirectory: {subdir_path}")
        
        generate_synthetic_data(main_dir, subdirs)
    
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

def generate_synthetic_data(main_dir, subdirs):
    """Generate synthetic data for testing the models"""
    print("Generating synthetic data for testing...")
    
    for subdir in subdirs:
        subdir_path = os.path.join(main_dir, subdir)
        num_files = 30 if subdir == "workload_A" else (40 if subdir == "workload_B" else 50)
        
        for i in range(num_files):
            file_path = os.path.join(subdir_path, f"sample_{i}.json")
            
            data = {
                "programming_details": {
                    "Nodes": [],
                    "Edges": [],
                    "Schedules": []
                },
                "scheduling_data": []
            }
            
            if subdir == "workload_A":
                num_nodes = random.randint(5, 15)
                exec_time_base = 100 + i * 10
            elif subdir == "workload_B":
                num_nodes = random.randint(15, 30)
                exec_time_base = 200 + i**1.5
            else:
                num_nodes = random.randint(30, 50)
                exec_time_base = 300 + i**2
            
            noise_factor = random.uniform(0.8, 1.2)
            execution_time = exec_time_base * noise_factor
            
            for j in range(num_nodes):
                op_types = ["add", "mul", "div", "sub", "conv", "relu", "sigmoid", "tanh"]
                op_hist = []
                
                for op in random.sample(op_types, min(4, len(op_types))):
                    count = random.randint(1, 10)
                    op_hist.append(f"{op}: {count}")
                
                node = {
                    "Name": f"Node_{j}",
                    "Details": {
                        "Op histogram": op_hist
                    }
                }
                data["programming_details"]["Nodes"].append(node)
            
            num_edges = min(num_nodes * 2, num_nodes * (num_nodes - 1) // 2)
            for j in range(num_edges):
                from_node = random.randint(0, num_nodes - 1)
                to_node = random.randint(0, num_nodes - 1)
                
                while to_node == from_node:
                    to_node = random.randint(0, num_nodes - 1)
                
                edge = {
                    "From": f"Node_{from_node}",
                    "To": f"Node_{to_node}",
                    "Name": f"Edge_{j}"
                }
                data["programming_details"]["Edges"].append(edge)
            
            bytes_at_production = random.randint(1000, 100000)
            num_vectors = random.randint(10, 1000)
            inner_parallelism = random.randint(1, 8)
            outer_parallelism = random.randint(1, 4)
            working_set = bytes_at_production * random.uniform(0.5, 2.0)
            
            schedule = {
                "Name": "total_execution_time_ms",
                "value": execution_time,
                "Details": {
                    "scheduling_feature": {
                        "bytes_at_production": bytes_at_production,
                        "bytes_at_realization": bytes_at_production * 0.8,
                        "bytes_at_root": bytes_at_production * 0.5,
                        "bytes_at_task": bytes_at_production * 0.3,
                        "inner_parallelism": inner_parallelism,
                        "outer_parallelism": outer_parallelism,
                        "num_productions": random.randint(5, 50),
                        "num_realizations": random.randint(5, 50),
                        "num_scalars": random.randint(10, 100),
                        "num_vectors": num_vectors,
                        "points_computed_total": random.randint(1000, 10000),
                        "working_set": working_set
                    }
                }
            }
            data["scheduling_data"].append(schedule)
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Generated synthetic data file: {file_path}")
    
    print(f"Synthetic data generation complete for {len(subdirs)} workloads")

def clean_and_transform_features(train_features, test_features):
    all_features_df = pd.DataFrame(train_features + test_features)
    
    # Impute missing values with median for numeric columns first
    numeric_cols = all_features_df.select_dtypes(include=['number']).columns
    all_features_df[numeric_cols] = all_features_df[numeric_cols].fillna(all_features_df[numeric_cols].median())
    
    # Outlier detection
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outliers = iso_forest.fit_predict(all_features_df.select_dtypes(include=['number']))
    all_features_df = all_features_df[outliers == 1]
    
    # Drop constant columns
    constant_columns = [col for col in all_features_df.columns 
                       if col != 'execution_time' and all_features_df[col].nunique() <= 1]
    all_features_df = all_features_df.drop(columns=constant_columns)
    print(f"Dropped {len(constant_columns)} constant columns")
    
    # Apply log transformation to execution time
    if 'execution_time' in all_features_df.columns:
        all_features_df['execution_time_log'] = np.log1p(all_features_df['execution_time'])
    
    # Polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly_cols = [col for col in all_features_df.columns if col not in ['execution_time', 'execution_time_log']]
    if poly_cols:
        poly_features = poly.fit_transform(all_features_df[poly_cols])
        poly_feature_names = poly.get_feature_names_out(poly_cols)
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=all_features_df.index)
        all_features_df = all_features_df.drop(columns=poly_cols).join(poly_df)
    
    # Clip extreme values
    for col in all_features_df.select_dtypes(include=['number']).columns:
        all_features_df[col] = all_features_df[col].clip(lower=all_features_df[col].quantile(0.01),
                                                        upper=all_features_df[col].quantile(0.99))
    
    # Remove highly correlated features
    corr_matrix = all_features_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]
    all_features_df = all_features_df.drop(columns=to_drop)
    print(f"Dropped {len(to_drop)} highly correlated columns")
    
    train_size = len(train_features)
    train_df = all_features_df.iloc[:train_size]
    test_df = all_features_df.iloc[train_size:]
    
    return train_df, test_df

def prepare_data_for_model(train_features, test_features):
    if not train_features or not test_features:
        raise ValueError("Empty training or test features provided")
    
    train_df, test_df = clean_and_transform_features(train_features, test_features)
    
    if 'execution_time_log' in train_df.columns:
        y_train = train_df['execution_time_log'].values.reshape(-1, 1)
        y_test = test_df['execution_time_log'].values.reshape(-1, 1)
        train_df = train_df.drop(['execution_time', 'execution_time_log'], axis=1, errors='ignore')
        test_df = test_df.drop(['execution_time', 'execution_time_log'], axis=1, errors='ignore')
        is_log_transformed = True
    else:
        y_train = train_df['execution_time'].values.reshape(-1, 1)
        y_test = test_df['execution_time'].values.reshape(-1, 1)
        train_df = train_df.drop('execution_time', axis=1, errors='ignore')
        test_df = test_df.drop('execution_time', axis=1, errors='ignore')
        is_log_transformed = False
    
    print("\nDebugging target values in prepare_data_for_model:")
    print(f"First 5 y_train raw: {y_train[:5].flatten()}")
    print(f"First 5 y_test raw: {y_test[:5].flatten()}")
    
    # Feature selection
    selector = SelectKBest(score_func=f_regression, k=min(30, train_df.shape[1]))
    X_train_selected = selector.fit_transform(train_df, y_train.ravel())
    X_test_selected = selector.transform(test_df)
    
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train_selected)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(X_test_selected)
    y_test_scaled = scaler_y.transform(y_test)
    
    print(f"First 5 y_train scaled: {y_train_scaled[:5].flatten()}")
    print(f"First 5 y_test scaled: {y_test_scaled[:5].flatten()}")
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    print(f"Input feature dimension: {X_train_scaled.shape[1]}")
    
    return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
            scaler_y, X_train_scaled.shape[1], is_log_transformed)

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=400, patience=75, model_name="model"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    
    # Learning rate warmup
    warmup_epochs = 10
    initial_lr = optimizer.param_groups[0]['lr'] / 10
    
    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            lr = initial_lr + (optimizer.param_groups[0]['lr'] - initial_lr) * epoch / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
            
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
        
        scheduler.step()
        
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
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss over Epochs - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loss_{model_name}.png')
    plt.close()
    print(f"Training plot saved as 'loss_{model_name}.png'")
    
    return train_losses, val_losses

def evaluate_model(model, X_test, y_test, y_scaler, file_names_test, is_log_transformed=False, original_execution_times=None, model_name="model"):
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
    
    print(f"\nDebugging transformed values before inverse log for {model_name}:")
    for i in range(min(5, len(y_test_transformed))):
        print(f"Sample {i}: y_test_transformed={y_test_transformed[i][0]}, y_pred_transformed={y_pred_transformed[i][0]}")
    
    if is_log_transformed:
        y_test_actual = np.expm1(y_test_transformed)
        y_pred_actual = np.expm1(y_pred_transformed)
    else:
        y_test_actual = y_test_transformed
        y_pred_actual = y_pred_transformed
    
    print(f"\nDebugging final values after all transformations for {model_name}:")
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
        print(f"\nResults for {subfolder} ({model_name}):")
        for result in results:
            print(f"File: {result['file']}")
            print(f"  Actual execution time: {result['actual']:.2f} ms")
            print(f"  Predicted execution time: {result['predicted']:.2f} ms")
            print(f"  Error percentage: {result['error_percentage']:.2f}%")
    
    mse = np.mean((y_test_actual - y_pred_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_actual - y_pred_actual))
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / (y_test_actual + 1e-8))) * 100
    
    print(f"\nOverall Model Performance ({model_name}):")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Residual analysis
    residuals = y_test_actual - y_pred_actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_actual, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Execution Time')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot - {model_name}')
    plt.grid(True)
    plt.savefig(f'residuals_{model_name}.png')
    plt.close()
    print(f"Residual plot saved as 'residuals_{model_name}.png'")
    
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
    
    X_train, y_train, X_test, y_test, y_scaler, input_size, is_log_transformed = prepare_data_for_model(train_features, test_features)
    
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32)
    
    # Train and evaluate LSTM model
    lstm_model = EnhancedLSTMModel(
        input_size=input_size,
        hidden_sizes=[512, 256, 128, 64],
        output_size=1,
        dropout_rate=0.3
    )
    
    lstm_criterion = nn.HuberLoss(delta=0.5)
    lstm_optimizer = optim.AdamW(lstm_model.parameters(), lr=0.00005, weight_decay=1e-4)
    
    print("\nBuilding and training Enhanced LSTM model...")
    lstm_train_losses, lstm_val_losses = train_model(
        lstm_model, 
        train_loader, 
        test_loader, 
        lstm_criterion, 
        lstm_optimizer, 
        num_epochs=400,
        patience=75,
        model_name="lstm_model"
    )
    
    print("\nEvaluating LSTM model:")
    lstm_y_test_actual, lstm_y_pred_actual = evaluate_model(
        lstm_model, X_test, y_test, y_scaler, test_file_names, 
        is_log_transformed, original_execution_times, model_name="lstm_model"
    )
    
    # Train and evaluate Transformer model
    transformer_model = EnhancedTransformerModel(
        input_size=input_size,
        hidden_size=256,
        num_heads=8,
        num_layers=4,
        dropout_rate=0.3,
        output_size=1
    )
    
    transformer_criterion = nn.HuberLoss(delta=0.5)
    transformer_optimizer = optim.AdamW(transformer_model.parameters(), lr=0.00005, weight_decay=1e-4)
    
    print("\nBuilding and training Enhanced Transformer model...")
    transformer_train_losses, transformer_val_losses = train_model(
        transformer_model, 
        train_loader, 
        test_loader, 
        transformer_criterion, 
        transformer_optimizer, 
        num_epochs=400,
        patience=75,
        model_name="transformer_model"
    )
    
    print("\nEvaluating Transformer model:")
    transformer_y_test_actual, transformer_y_pred_actual = evaluate_model(
        transformer_model, X_test, y_test, y_scaler, test_file_names, 
        is_log_transformed, original_execution_times, model_name="transformer_model"
    )
    
    # Ensemble model
    print("\nEvaluating Ensemble model...")
    ensemble_model = EnsembleModel(lstm_model, transformer_model, weight_lstm=0.4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        ensemble_pred_scaled = ensemble_model.predict(X_test, device)
    
    ensemble_pred_scaled = ensemble_pred_scaled.cpu().numpy()
    ensemble_y_test_actual = y_scaler.inverse_transform(y_test.cpu().numpy())
    ensemble_y_pred_actual = y_scaler.inverse_transform(ensemble_pred_scaled)
    
    if is_log_transformed:
        ensemble_y_test_actual = np.expm1(ensemble_y_test_actual)
        ensemble_y_pred_actual = np.expm1(ensemble_y_pred_actual)
    
    results_by_subfolder = {}
    for i, file_path in enumerate(test_file_names):
        subfolder = file_path.split('/')[0]
        if subfolder not in results_by_subfolder:
            results_by_subfolder[subfolder] = []
        
        actual_val = ensemble_y_test_actual[i][0]
        pred_val = ensemble_y_pred_actual[i][0]
        error_percentage = abs(actual_val - pred_val) / actual_val * 100 if actual_val > 0 else 0
        
        results_by_subfolder[subfolder].append({
            'file': file_path,
            'actual': actual_val,
            'predicted': pred_val,
            'error_percentage': error_percentage
        })
    
    for subfolder, results in results_by_subfolder.items():
        print(f"\nResults for {subfolder} (ensemble_model):")
        for result in results:
            print(f"File: {result['file']}")
            print(f"  Actual execution time: {result['actual']:.2f} ms")
            print(f"  Predicted execution time: {result['predicted']:.2f} ms")
            print(f"  Error percentage: {result['error_percentage']:.2f}%")
    
    mse = np.mean((ensemble_y_test_actual - ensemble_y_pred_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(ensemble_y_test_actual - ensemble_y_pred_actual))
    mape = np.mean(np.abs((ensemble_y_test_actual - ensemble_y_pred_actual) / (ensemble_y_test_actual + 1e-8))) * 100
    
    print(f"\nOverall Model Performance (ensemble_model):")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Save models
    print("\nSaving trained models...")
    device = torch.device('cpu')
    
    try:
        lstm_model.eval()
        lstm_model.to(device)
        lstm_sample_input = torch.randn(1, 1, input_size).to(device)
        lstm_traced_model = torch.jit.trace(lstm_model, lstm_sample_input)
        lstm_traced_model.save("lstm_model.pt")
        print("LSTM model successfully saved as 'lstm_model.pt'")
    except Exception as e:
        print(f"Error saving LSTM model: {str(e)}")
    
    try:
        transformer_model.eval()
        transformer_model.to(device)
        torch.save(transformer_model.state_dict(), "transformer_model.pth")
        print("Transformer model successfully saved as 'transformer_model.pth'")
        print("Note: To load the Transformer model, instantiate EnhancedTransformerModel with the same parameters and load the state_dict:")
        print("  model = EnhancedTransformerModel(input_size=<input_size>, hidden_size=256, num_heads=8, num_layers=4, dropout_rate=0.3, output_size=1)")
        print("  model.load_state_dict(torch.load('transformer_model.pth'))")
    except Exception as e:
        print(f"Error saving Transformer model: {str(e)}")
    
    return {
        'lstm_model': lstm_model,
        'transformer_model': transformer_model,
        'ensemble_model': ensemble_model,
        'y_scaler': y_scaler,
        'lstm_y_test_actual': lstm_y_test_actual,
        'lstm_y_pred_actual': lstm_y_pred_actual,
        'transformer_y_test_actual': transformer_y_test_actual,
        'transformer_y_pred_actual': transformer_y_pred_actual,
        'ensemble_y_test_actual': ensemble_y_test_actual,
        'ensemble_y_pred_actual': ensemble_y_pred_actual
    }

if __name__ == "__main__":
    main_dir = "synthetic_data"
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    result = main(main_dir)
    if result is not None:
        lstm_model = result['lstm_model']
        transformer_model = result['transformer_model']
        ensemble_model = result['ensemble_model']
        y_scaler = result['y_scaler']
        lstm_y_test_actual = result['lstm_y_test_actual/'
