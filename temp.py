import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

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
        'scheduling_count': len(scheduling_features),
        'total_bytes_at_production': 0.0,
        'total_vectors': 0.0,
        'total_parallelism': 0.0
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
    
    features['bytes_per_parallelism'] = features['total_bytes_at_production'] / (features['total_parallelism'] + 1e-8)
    if 'nodes_count' in features and 'scheduling_count' in features:
        features['nodes_per_schedule'] = features['nodes_count'] / (features['scheduling_count'] + 1e-8)
    
    features['inner_parallelism_total_parallelism'] = features.get('sched_inner_parallelism', 0) * features['total_parallelism']
    
    # Refined computation_efficiency: (points computed / (bytes * parallelism)) to account for parallel processing
    if 'sched_points_computed_total' in features and features['total_bytes_at_production'] > 0 and features['total_parallelism'] > 0:
        features['computation_efficiency'] = features['sched_points_computed_total'] / (features['total_bytes_at_production'] * features['total_parallelism'])
    else:
        features['computation_efficiency'] = 0.0
    
    # Additional interaction terms
    features['comp_efficiency_total_vectors'] = features['computation_efficiency'] * features['total_vectors']
    features['scheduling_count_total_vectors'] = features['scheduling_count'] * features['total_vectors']
    features['sched_inner_parallelism_squared'] = features.get('sched_inner_parallelism', 0) ** 2
    features['computation_efficiency_squared'] = features['computation_efficiency'] ** 2
    
    if execution_time > 0:
        features['bytes_processing_rate'] = features['total_bytes_at_production'] / execution_time
    else:
        features['bytes_processing_rate'] = 0.0
    
    if 'sched_working_set' in features and features['total_bytes_at_production'] > 0:
        features['memory_utilization_ratio'] = features['sched_working_set'] / features['total_bytes_at_production']
    else:
        features['memory_utilization_ratio'] = 0.0
    
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

def clean_and_transform_features(train_features, test_features, test_size=50):
    all_features_df = pd.DataFrame(train_features + test_features)
    
    Q1 = all_features_df['execution_time'].quantile(0.25)
    Q3 = all_features_df['execution_time'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2.0 * IQR
    upper_bound = Q3 + 2.0 * IQR
    all_features_df = all_features_df[(all_features_df['execution_time'] >= lower_bound) & (all_features_df['execution_time'] <= upper_bound)]
    print(f"Removed {len(train_features + test_features) - len(all_features_df)} outliers based on execution time")
    
    if len(all_features_df) < test_size:
        raise ValueError(f"After outlier removal, only {len(all_features_df)} samples remain, but {test_size} are required for the test set.")
    
    train_size = len(all_features_df) - test_size
    if train_size <= 0:
        raise ValueError("Not enough samples remaining for training after outlier removal and reserving test set.")
    
    low_importance_features = [
        'sched_num_scalars', 'sched_points_computed_total', 'sched_bytes_at_realization',
        'sched_outer_parallelism', 'sched_num_realizations', 'sched_num_productions',
        'sched_bytes_at_root', 'sched_bytes_at_production', 'op_cast', 'op_eq', 'op_ne',
        'op_or', 'op_and', 'op_le', 'op_lt', 'op_not'
    ]
    all_features_df = all_features_df.drop(columns=[col for col in low_importance_features if col in all_features_df.columns])
    print(f"Dropped {len(low_importance_features)} low-importance features")
    
    skewed_features = ['total_bytes_at_production', 'total_vectors', 'total_parallelism', 'computation_efficiency', 'bytes_processing_rate']
    for feature in skewed_features:
        if feature in all_features_df.columns:
            # Apply robust normalization before log transform
            scaler = RobustScaler()
            all_features_df[feature] = scaler.fit_transform(all_features_df[[feature]])
            all_features_df[f'log_{feature}'] = np.log1p(all_features_df[feature].clip(lower=0))
            all_features_df = all_features_df.drop(columns=[feature])
    
    all_features_df = all_features_df.fillna(0)
    
    constant_columns = [col for col in all_features_df.columns 
                       if col != 'execution_time' and all_features_df[col].nunique() == 1]
    all_features_df = all_features_df.drop(columns=constant_columns)
    print(f"Dropped {len(constant_columns)} constant columns")
    
    corr_matrix = all_features_df.drop(['execution_time'], axis=1).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    all_features_df = all_features_df.drop(columns=to_drop)
    print(f"Dropped {len(to_drop)} highly correlated features")
    
    if 'log_total_bytes_at_production' in all_features_df.columns and 'log_total_vectors' in all_features_df.columns:
        all_features_df['log_bytes_per_vector'] = all_features_df['log_total_bytes_at_production'] / (all_features_df['log_total_vectors'] + 1e-8)
    else:
        print("Warning: 'log_total_bytes_at_production' or 'log_total_vectors' not found in DataFrame, skipping log_bytes_per_vector calculation")
        all_features_df['log_bytes_per_vector'] = 0.0
    
    numeric_cols = all_features_df.select_dtypes(include=['number']).columns
    all_features_df = all_features_df[numeric_cols]
    
    train_df = all_features_df.iloc[:train_size]
    test_df = all_features_df.iloc[train_size:]
    
    if len(test_df) != test_size:
        raise ValueError(f"Test set has {len(test_df)} samples, but expected {test_size}.")
    
    return train_df, test_df

def prepare_data_for_model(train_features, test_features, test_size=50):
    train_df, test_df = clean_and_transform_features(train_features, test_features, test_size=test_size)
    
    y_train = np.log1p(train_df['execution_time'].values.reshape(-1, 1))
    y_test = np.log1p(test_df['execution_time'].values.reshape(-1, 1))
    train_df = train_df.drop('execution_time', axis=1)
    test_df = test_df.drop('execution_time', axis=1)
    is_log_transformed = True
    
    print("\nDebugging target values in prepare_data_for_model:")
    print(f"First 5 y_train raw (log-transformed): {y_train[:5].flatten()}")
    print(f"First 5 y_test raw (log-transformed): {y_test[:5].flatten()}")
    
    scaler_X = StandardScaler()
    scaler_y = RobustScaler()
    
    # Apply SMOTE to balance the training data
    X_train = train_df.to_numpy()
    y_train_labels = np.digitize(np.expm1(y_train.flatten()), bins=[0, 100, 500, np.inf])
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_balanced, y_train_labels_balanced = smote.fit_resample(X_train, y_train_labels)
    
    # Map balanced labels back to y_train values
    y_train_balanced = np.zeros((len(y_train_labels_balanced), 1))
    for i, label in enumerate(y_train_labels_balanced):
        indices = np.where(y_train_labels == label)[0]
        if len(indices) > 0:
            y_train_balanced[i] = y_train[indices[0]]
        else:
            y_train_balanced[i] = y_train[i % len(y_train)]
    
    X_train_scaled = scaler_X.fit_transform(X_train_balanced)
    y_train_scaled = scaler_y.fit_transform(y_train_balanced)
    X_test_scaled = scaler_X.transform(test_df)
    y_test_scaled = scaler_y.transform(y_test)
    
    print(f"First 5 y_train scaled: {y_train_scaled[:5].flatten()}")
    print(f"First 5 y_test scaled: {y_test_scaled[:5].flatten()}")
    
    X_train_scaled_aug = []
    y_train_scaled_aug = []
    train_df_array = X_train_scaled
    
    for i in range(len(y_train_scaled)):
        actual_time = np.expm1(scaler_y.inverse_transform(y_train_scaled[i].reshape(-1, 1))[0][0])
        X_train_scaled_aug.append(X_train_scaled[i])
        y_train_scaled_aug.append(y_train_scaled[i])
        
        inner_parallelism_idx = train_df.columns.get_loc('sched_inner_parallelism') if 'sched_inner_parallelism' in train_df.columns else -1
        total_parallelism_idx = train_df.columns.get_loc('total_parallelism') if 'total_parallelism' in train_df.columns else -1
        comp_efficiency_idx = train_df.columns.get_loc('log_computation_efficiency') if 'log_computation_efficiency' in train_df.columns else -1
        
        is_significant = False
        if inner_parallelism_idx != -1 and train_df_array[i, inner_parallelism_idx] > np.percentile(train_df_array[:, inner_parallelism_idx], 75):
            is_significant = True
        if total_parallelism_idx != -1 and train_df_array[i, total_parallelism_idx] > np.percentile(train_df_array[:, total_parallelism_idx], 75):
            is_significant = True
        if comp_efficiency_idx != -1 and train_df_array[i, comp_efficiency_idx] > np.percentile(train_df_array[:, comp_efficiency_idx], 75):
            is_significant = True
        
        augment_count = 0
        if actual_time < 100:
            augment_count = 7  # Further increased for small execution times
        elif actual_time > 500 or is_significant:
            augment_count = 5
        else:
            augment_count = 3
        
        for _ in range(augment_count):
            noise_X = np.random.normal(0, 0.03, X_train_scaled[i].shape)
            noise_y = np.random.normal(0, 0.03, y_train_scaled[i].shape)
            X_train_scaled_aug.append(X_train_scaled[i] + noise_X)
            y_train_scaled_aug.append(y_train_scaled[i] + noise_y)
    
    X_train_scaled_aug = np.array(X_train_scaled_aug)
    y_train_scaled_aug = np.array(y_train_scaled_aug)
    
    X_train_tensor = torch.FloatTensor(X_train_scaled_aug).unsqueeze(1)
    y_train_tensor = torch.FloatTensor(y_train_scaled_aug)
    X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    input_size = X_train_scaled.shape[1]
    print(f"Input feature dimension: {input_size}")
    
    return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
            scaler_y, input_size, is_log_transformed, train_df.columns)

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / (self.v.size(0) ** 0.5)
        self.v.data.uniform_(-stdv, stdv)
    
    def forward(self, hidden, encoder_outputs):
        batch_size, seq_len, _ = encoder_outputs.size()
        hidden = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attention(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy @ self.v
        attention_weights = torch.softmax(energy, dim=1).unsqueeze(2)
        context = (attention_weights * encoder_outputs).sum(dim=1)
        return context

class AdvancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=1, dropout_rate=0.15):
        super(AdvancedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 2
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=self.num_layers, batch_first=True, dropout=dropout_rate)
        self.attention = Attention(hidden_size)
        
        self.fc1 = nn.Linear(hidden_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, output_size)
        
        self.residual = nn.Linear(hidden_size, 32)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        context = self.attention(hn, lstm_out)
        
        x = context
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        residual = self.residual(context)
        x = self.fc3(x)
        x = self.bn3(x)
        x += residual
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        return x

def create_data_loaders(X_train, y_train, batch_size=16):
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader

class AdvancedLoss(nn.Module):
    def __init__(self, mape_weight=0.3, mse_weight=0.5, quantile_weight=0.2, epsilon=1e-4, feature_indices=None, feature_importances=None):
        super(AdvancedLoss, self).__init__()
        self.mape_weight = mape_weight
        self.mse_weight = mse_weight
        self.quantile_weight = quantile_weight
        self.epsilon = epsilon
        self.mse_loss = nn.MSELoss(reduction='none')
        self.feature_indices = feature_indices if feature_indices else {}
        self.feature_importances = feature_importances if feature_importances else {}
        self.quantile = 0.5  # Median
    
    def quantile_loss(self, outputs, targets):
        errors = targets - outputs
        loss = torch.where(errors >= 0, self.quantile * errors, (self.quantile - 1) * errors)
        return loss.mean()
    
    def forward(self, outputs, targets, inputs):
        mse = self.mse_loss(outputs, targets)
        mape = torch.abs((targets - outputs) / (targets + self.epsilon)) * 100
        
        weights = torch.ones_like(targets)
        weights = torch.where(targets < -1.0, 3.0, weights)  # Higher weight for small execution times
        weights = torch.where((targets >= -1.0) & (targets <= 2.0), 1.5, weights)
        weights = torch.where(targets > 2.0, 1.2, weights)
        
        for feature, idx in self.feature_indices.items():
            if idx != -1 and feature in self.feature_importances:
                feature_vals = inputs[:, 0, idx]
                importance = self.feature_importances[feature]
                weights = torch.where(
                    feature_vals > 1.0,
                    weights * (1.0 + importance * 2.5),
                    weights
                )
        
        weighted_mse = (mse * weights).mean()
        weighted_mape = (mape * weights).mean()
        quantile_loss = self.quantile_loss(outputs, targets)
        
        return (self.mape_weight * weighted_mape +
                self.mse_weight * weighted_mse +
                self.quantile_weight * quantile_loss)

def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=1000, patience=100, fold=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets, inputs)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        scheduler.step()
        
        print(f'Fold {fold+1}, Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}')
        
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model state for this fold.")
    
    return train_losses

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
    error_by_range = {'small (<100ms)': [], 'medium (100-500ms)': [], 'large (>500ms)': []}
    
    for i, file_path in enumerate(file_names_test):
        subfolder = file_path.split('/')[0]
        if subfolder not in results_by_subfolder:
            results_by_subfolder[subfolder] = []
        
        actual_val = y_test_actual[i][0]
        pred_val = y_pred_actual[i][0]
        error_percentage = abs(actual_val - pred_val) / actual_val * 100 if actual_val > 0 else 0
        error_percentage = min(error_percentage, 1000.0)
        
        if actual_val < 100:
            error_by_range['small (<100ms)'].append(error_percentage)
        elif 100 <= actual_val <= 500:
            error_by_range['medium (100-500ms)'].append(error_percentage)
        else:
            error_by_range['large (>500ms)'].append(error_percentage)
        
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
        mdape = np.median(np.abs((y_test_actual[mask] - y_pred_actual[mask]) / y_test_actual[mask])) * 100
    else:
        mape = 0.0
        mdape = 0.0
    
    print("\nError Analysis by Execution Time Range:")
    for range_name, errors in error_by_range.items():
        if errors:
            avg_error = np.mean(errors)
            print(f"{range_name}: Average Error = {avg_error:.2f}% (n={len(errors)})")
        else:
            print(f"{range_name}: No samples")
    
    print("\nOverall Model Performance:")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape:.2f}%")
    print(f"MdAPE: {mdape:.2f}%")
    
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
    
    X_train, y_train, X_test, y_test, y_scaler, input_size, is_log_transformed, feature_columns = prepare_data_for_model(train_features, test_features, test_size=50)
    
    # Define feature importances from the Random Forest report
    feature_importances = {
        'computation_efficiency': 0.6064,
        'sched_inner_parallelism': 0.2135,
        'total_parallelism': 0.0038,
        'total_vectors': 0.0138,
        'scheduling_count': 0.0454,
        'total_bytes_at_production': 0.0357,
        'bytes_processing_rate': 0.0064
    }
    
    # Map feature indices
    feature_indices = {}
    for feature in feature_importances.keys():
        log_feature = f'log_{feature}' if feature in ['computation_efficiency', 'total_bytes_at_production', 'total_vectors', 'total_parallelism', 'bytes_processing_rate'] else feature
        if log_feature in feature_columns:
            feature_indices[feature] = feature_columns.get_loc(log_feature)
        else:
            feature_indices[feature] = feature_columns.get_loc(feature) if feature in feature_columns else -1
    
    criterion = AdvancedLoss(
        mape_weight=0.3,
        mse_weight=0.5,
        quantile_weight=0.2,
        epsilon=1e-4,
        feature_indices=feature_indices,
        feature_importances=feature_importances
    )
    
    # K-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    X_train_np = X_train.numpy().squeeze(1)
    y_train_np = y_train.numpy()
    
    fold_models = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_np)):
        print(f"\nTraining Fold {fold+1}/5")
        
        X_train_fold = torch.FloatTensor(X_train_np[train_idx]).unsqueeze(1)
        y_train_fold = torch.FloatTensor(y_train_np[train_idx])
        
        train_loader = create_data_loaders(X_train_fold, y_train_fold, batch_size=16)
        
        model = AdvancedLSTMModel(
            input_size=input_size,
            hidden_size=256,
            output_size=1,
            dropout_rate=0.15
        )
        optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        
        train_losses = train_model(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            num_epochs=1000,
            patience=100,
            fold=fold
        )
        
        fold_models.append(model)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss over Epochs (Fold {fold+1})')
        plt.legend()
        plt.grid(True)
        try:
            plt.savefig(f'loss_fold_{fold+1}.png')
            plt.close()
            print(f"Training plot for fold {fold+1} saved as 'loss_fold_{fold+1}.png'")
        except (OSError, RuntimeError) as e:
            print(f"Warning: Failed to save training plot: {str(e)}")
            plt.close()
    
    print("\nEvaluating ensemble model on test set:")
    predictions = []
    for model in fold_models:
        model.eval()
        with torch.no_grad():
            pred = model(X_test)
        predictions.append(pred.cpu().numpy())
    
    y_pred_avg = np.mean(predictions, axis=0)
    y_test_np = y_test.numpy()
    
    y_test_transformed = y_scaler.inverse_transform(y_test_np)
    y_pred_transformed = y_scaler.inverse_transform(y_pred_avg)
    
    if is_log_transformed:
        y_test_actual = np.expm1(y_test_transformed)
        y_pred_actual = np.expm1(y_pred_transformed)
    else:
        y_test_actual = y_test_transformed
        y_pred_actual = y_pred_transformed
    
    y_pred_actual = np.maximum(y_pred_actual, 1e-2)
    
    results_by_subfolder = {}
    error_by_range = {'small (<100ms)': [], 'medium (100-500ms)': [], 'large (>500ms)': []}
    
    for i, file_path in enumerate(test_file_names):
        subfolder = file_path.split('/')[0]
        if subfolder not in results_by_subfolder:
            results_by_subfolder[subfolder] = []
        
        actual_val = y_test_actual[i][0]
        pred_val = y_pred_actual[i][0]
        error_percentage = abs(actual_val - pred_val) / actual_val * 100 if actual_val > 0 else 0
        error_percentage = min(error_percentage, 1000.0)
        
        if actual_val < 100:
            error_by_range['small (<100ms)'].append(error_percentage)
        elif 100 <= actual_val <= 500:
            error_by_range['medium (100-500ms)'].append(error_percentage)
        else:
            error_by_range['large (>500ms)'].append(error_percentage)
        
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
        mdape = np.median(np.abs((y_test_actual[mask] - y_pred_actual[mask]) / y_test_actual[mask])) * 100
    else:
        mape = 0.0
        mdape = 0.0
    
    print("\nError Analysis by Execution Time Range:")
    for range_name, errors in error_by_range.items():
        if errors:
            avg_error = np.mean(errors)
            print(f"{range_name}: Average Error = {avg_error:.2f}% (n={len(errors)})")
        else:
            print(f"{range_name}: No samples")
    
    print("\nOverall Model Performance (Ensemble):")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape:.2f}%")
    print(f"MdAPE: {mdape:.2f}%")
    
    print("\nSaving the best model from fold 1 as 'lstm_model.pt'...")
    best_model = fold_models[0]
    best_model.eval()
    device = next(best_model.parameters()).device
    print(f"Model is on device: {device}")
    
    try:
        sample_input = torch.randn(1, 1, input_size).to(device)
        traced_model = torch.jit.trace(best_model, sample_input)
        traced_model.save("lstm_model.pt")
        print("Model successfully saved as 'lstm_model.pt'")
    except (OSError, RuntimeError) as e:
        print(f"Error saving the model: {str(e)}")
    
    return best_model, y_scaler, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "synthetic_data"
    random.seed(42)
    result = main(main_dir)
    if result is not None:
        model, y_scaler, y_test_actual, y_pred_actual = result
