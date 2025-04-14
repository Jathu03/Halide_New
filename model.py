import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union

# [EnhancedLSTMModel: Modified to include bidirectional LSTM and layer normalization]
class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int] = [512, 256, 128, 64], 
                 output_size: int = 1, dropout_rate: float = 0.3):
        super(EnhancedLSTMModel, self).__init__()
        
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        self.lstm_layers.append(nn.LSTM(input_size, hidden_sizes[0], batch_first=True, bidirectional=True))
        self.norm_layers.append(nn.LayerNorm(hidden_sizes[0] * 2))
        self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        for i in range(1, len(hidden_sizes)):
            self.lstm_layers.append(nn.LSTM(hidden_sizes[i-1] * 2, hidden_sizes[i], batch_first=True, bidirectional=True))
            self.norm_layers.append(nn.LayerNorm(hidden_sizes[i] * 2))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        self.attention = nn.Linear(hidden_sizes[-1] * 2, 1)
        
        self.fc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
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
        for i, (lstm, norm, dropout) in enumerate(zip(self.lstm_layers, self.norm_layers, self.dropout_layers)):
            lstm_out, _ = lstm(lstm_out)
            lstm_out = norm(lstm_out)
            if i < len(self.lstm_layers) - 1:
                lstm_out = dropout(lstm_out)
        
        attn_output = self.attention_net(lstm_out)
        
        fc_out = self.fc_layers[0](attn_output)
        fc_out = self.bn_layers[0](fc_out)
        fc_out = self.leaky_relu(fc_out)
        
        residual = fc_out
        if not self.has_residual:
            residual = self.residual_adapter(residual)
        
        fc_out = self.fc_layers[1](fc_out)
        fc_out = self.bn_layers[1](fc_out)
        fc_out = self.leaky_relu(fc_out)
        
        fc_out = fc_out + residual
        
        output = self.output_layer(fc_out)
        
        return output

# [EnhancedTransformerModel: Added positional encoding and more layers]
class EnhancedTransformerModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256, num_heads: int = 8, 
                 num_layers: int = 4, dropout_rate: float = 0.3, output_size: int = 1):
        super(EnhancedTransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        self.pos_encoder = PositionalEncoding(hidden_size, dropout_rate)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.global_attention = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
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
        batch_size, seq_len, _ = x.size()
        
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        transformer_output = self.transformer_encoder(x)
        
        attention_weights = self.global_attention(transformer_output)
        context = torch.bmm(attention_weights.transpose(1, 2), transformer_output).squeeze(1)
        
        output = self.fc_layers(context)
        
        output = self.output_layer(output)
        
        return output

# [New PositionalEncoding class for Transformer]
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# [Modified extract_features_from_file: Added interaction terms and outlier clipping]
def extract_features_from_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    execution_time = get_execution_time(file_path)
    
    if execution_time is None:
        print(f"Warning: No execution time found in {file_path}")
        return None
    
    # Clip execution time to handle outliers
    execution_time = np.clip(execution_time, 1, 10000)
    
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
    
    # Interaction term
    features['nodes_edges_interaction'] = len(nodes_features) * len(edges_features)
    
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
                    # Clip large values
                    features[f'sched_{metric}'] = np.clip(scheduling_features[0][metric], 0, 1e9)
        
        total_bytes_at_production = sum(sf.get('bytes_at_production', 0) for sf in scheduling_features if isinstance(sf, dict))
        total_vectors = sum(sf.get('num_vectors', 0) for sf in scheduling_features if isinstance(sf, dict))
        total_parallelism = sum(sf.get('inner_parallelism', 0) * sf.get('outer_parallelism', 1) for sf in scheduling_features if isinstance(sf, dict))
        
        # Log transform large values
        features['total_bytes_at_production'] = np.log1p(total_bytes_at_production)
        features['total_vectors'] = np.log1p(total_vectors)
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

# [Modified generate_synthetic_data: More realistic distributions]
def generate_synthetic_data(main_dir, subdirs):
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
                num_nodes = random.randint(5, 20)
                exec_time_base = np.random.normal(150, 50)
            elif subdir == "workload_B":
                num_nodes = random.randint(15, 40)
                exec_time_base = np.random.normal(300, 100)
            else:
                num_nodes = random.randint(30, 60)
                exec_time_base = np.random.normal(500, 150)
            
            noise_factor = np.random.normal(1.0, 0.2)
            execution_time = max(1, exec_time_base * noise_factor)
            
            for j in range(num_nodes):
                op_types = ["add", "mul", "div", "sub", "conv", "relu", "sigmoid", "tanh"]
                op_hist = []
                
                for op in random.sample(op_types, min(5, len(op_types))):
                    count = int(np.random.exponential(5))
                    op_hist.append(f"{op}: {count}")
                
                node = {
                    "Name": f"Node_{j}",
                    "Details": {
                        "Op histogram": op_hist
                    }
                }
                data["programming_details"]["Nodes"].append(node)
            
            num_edges = min(num_nodes * 3, num_nodes * (num_nodes - 1) // 2)
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
            
            bytes_at_production = int(np.random.exponential(10000))
            num_vectors = int(np.random.exponential(100))
            inner_parallelism = random.randint(1, 16)
            outer_parallelism = random.randint(1, 8)
            working_set = bytes_at_production * np.random.uniform(0.5, 2.0)
            
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
                        "num_productions": random.randint(5, 100),
                        "num_realizations": random.randint(5, 100),
                        "num_scalars": random.randint(10, 200),
                        "num_vectors": num_vectors,
                        "points_computed_total": random.randint(1000, 20000),
                        "working_set": working_set
                    }
                }
            }
            data["scheduling_data"].append(schedule)
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Generated synthetic data file: {file_path}")
    
    print(f"Synthetic data generation complete for {len(subdirs)} workloads")

# [Modified clean_and_transform_features: Robust scaling and feature selection]
def clean_and_transform_features(train_features, test_features):
    all_features_df = pd.DataFrame(train_features + test_features)
    
    # Median imputation instead of zero-filling
    numeric_cols = all_features_df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        all_features_df[col] = all_features_df[col].fillna(all_features_df[col].median())
    
    # Log transform skewed features
    skewed_features = ['total_bytes_at_production', 'total_vectors', 'bytes_per_vector']
    for col in skewed_features:
        if col in all_features_df.columns:
            all_features_df[col] = np.log1p(all_features_df[col].clip(lower=0))
    
    # Clip outliers
    for col in numeric_cols:
        if col != 'execution_time':
            q1, q3 = all_features_df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            all_features_df[col] = all_features_df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Remove constant columns
    constant_columns = [col for col in all_features_df.columns 
                       if col != 'execution_time' and all_features_df[col].nunique() == 1]
    all_features_df = all_features_df.drop(columns=constant_columns)
    print(f"Dropped {len(constant_columns)} constant columns")
    
    # Feature selection using correlation
    if 'execution_time' in all_features_df.columns:
        X = all_features_df.drop(['execution_time'], axis=1)
        y = all_features_df['execution_time']
        selector = SelectKBest(score_func=f_regression, k=min(50, X.shape[1]))
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        selected_features.append('execution_time')
        all_features_df = all_features_df[selected_features]
        print(f"Selected {len(selected_features)-1} features based on correlation")
    
    # Log transform execution time
    all_features_df['execution_time_log'] = np.log1p(all_features_df['execution_time'].clip(lower=0))
    
    train_size = len(train_features)
    train_df = all_features_df.iloc[:train_size]
    test_df = all_features_df.iloc[train_size:]
    
    return train_df, test_df

# [Modified prepare_data_for_model: Use RobustScaler]
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
    
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    
    X_train_scaled = scaler_X.fit_transform(train_df)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(test_df)
    y_test_scaled = scaler_y.transform(y_test)
    
    print(f"First 5 y_train scaled: {y_train_scaled[:5].flatten()}")
    print(f"First 5 y_test scaled: {y_test_scaled[:5].flatten()}")
    
    # Add noise to training data for robustness
    noise = np.random.normal(0, 0.01, X_train_scaled.shape)
    X_train_scaled += noise
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    print(f"Input feature dimension: {X_train_scaled.shape[1]}")
    
    return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
            scaler_y, X_train_scaled.shape[1], is_log_transformed)

# [Modified train_model: Cosine annealing and combined loss]
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=200, patience=30, model_name="model"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    l1_lambda = 1e-5
    
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
            huber_loss = criterion(outputs, targets)
            
            # Add L1 loss
            l1_loss = 0
            for param in model.parameters():
                l1_loss += torch.norm(param, 1)
            loss = huber_loss + l1_lambda * l1_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += huber_loss.item() * inputs.size(0)
        
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

# [Modified evaluate_model: Ensemble prediction and clipping]
def evaluate_model(lstm_model, transformer_model, X_test, y_test, y_scaler, file_names_test, 
                   is_log_transformed=False, original_execution_times=None, model_name="ensemble"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm_model.to(device)
    transformer_model.to(device)
    lstm_model.eval()
    transformer_model.eval()
    
    X_test = X_test.to(device)
    with torch.no_grad():
        lstm_pred = lstm_model(X_test)
        transformer_pred = transformer_model(X_test)
        # Weighted ensemble: 60% LSTM, 40% Transformer
        y_pred_scaled = 0.6 * lstm_pred + 0.4 * transformer_pred
    
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
    
    # Clip negative predictions
    y_pred_actual = np.clip(y_pred_actual, 1, None)
    
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
    
    return y_test_actual, y_pred_actual

# [Modified main: Train both models and use ensemble]
def main(main_dir):
    print(f"Processing main directory: {main_dir}")
    train_features, test_features, test_file_names = process_main_directory(main_dir)
    
    print(f"Total training samples: {len(train_features)}")
    print(f"Total test samples: {len(test_features)}")
    
    if len(train_features) == 0 or len(test_features) == 0:
        print("Error: No valid training or test data found")
        return None
    
    original_execution_times = {}
    for feature, fname in zip(test_features, test_file_names):
        original_execution_times[fname] = feature['execution_time']
    
    X_train, y_train, X_test, y_test, y_scaler, input_size, is_log_transformed = prepare_data_for_model(train_features, test_features)
    
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=16)
    
    # Train LSTM model
    lstm_model = EnhancedLSTMModel(
        input_size=input_size,
        hidden_sizes=[512, 256, 128, 64],
        output_size=1,
        dropout_rate=0.3
    )
    
    lstm_criterion = nn.HuberLoss(delta=0.5)
    lstm_optimizer = optim.AdamW(lstm_model.parameters(), lr=0.0003, weight_decay=1e-4)
    
    print("\nBuilding and training Enhanced LSTM model...")
    lstm_train_losses, lstm_val_losses = train_model(
        lstm_model, 
        train_loader, 
        test_loader, 
        lstm_criterion, 
        lstm_optimizer, 
        num_epochs=300,
        patience=50,
        model_name="lstm_model"
    )
    
    # Train Transformer model
    transformer_model = EnhancedTransformerModel(
        input_size=input_size,
        hidden_size=256,
        num_heads=8,
        num_layers=4,
        dropout_rate=0.3,
        output_size=1
    )
    
    transformer_criterion = nn.HuberLoss(delta=0.5)
    transformer_optimizer = optim.AdamW(transformer_model.parameters(), lr=0.0003, weight_decay=1e-4)
    
    print("\nBuilding and training Enhanced Transformer model...")
    transformer_train_losses, transformer_val_losses = train_model(
        transformer_model, 
        train_loader, 
        test_loader, 
        transformer_criterion, 
        transformer_optimizer, 
        num_epochs=300,
        patience=50,
        model_name="transformer_model"
    )
    
    print("\nEvaluating ensemble model:")
    y_test_actual, y_pred_actual = evaluate_model(
        lstm_model, transformer_model, X_test, y_test, y_scaler, test_file_names, 
        is_log_transformed, original_execution_times, model_name="ensemble"
    )
    
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
    except Exception as e:
        print(f"Error saving Transformer model: {str(e)}")
    
    return {
        'lstm_model': lstm_model,
        'transformer_model': transformer_model,
        'y_scaler': y_scaler,
        'y_test_actual': y_test_actual,
        'y_pred_actual': y_pred_actual
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
        y_scaler = result['y_scaler']
        y_test_actual = result['y_test_actual']
        y_pred_actual = result['y_pred_actual']
