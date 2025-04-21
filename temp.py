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
        'total_bytes_at_production': 0.0,  # Default value
        'total_vectors': 0.0,              # Default value
        'total_parallelism': 0.0           # Default value
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
    
    # Add interaction features
    features['bytes_per_parallelism'] = features['total_bytes_at_production'] / (features['total_parallelism'] + 1e-8)
    if 'nodes_count' in features and 'scheduling_count' in features:
        features['nodes_per_schedule'] = features['nodes_count'] / (features['scheduling_count'] + 1e-8)
    
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
    
    # Remove outliers in execution time using IQR
    Q1 = all_features_df['execution_time'].quantile(0.25)
    Q3 = all_features_df['execution_time'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    all_features_df = all_features_df[(all_features_df['execution_time'] >= lower_bound) & (all_features_df['execution_time'] <= upper_bound)]
    print(f"Removed {len(train_features + test_features) - len(all_features_df)} outliers based on execution time")
    
    # After outlier removal, ensure we have enough samples for the test set
    if len(all_features_df) < test_size:
        raise ValueError(f"After outlier removal, only {len(all_features_df)} samples remain, but {test_size} are required for the test set.")
    
    # Adjust the train-test split based on the remaining data
    train_size = len(all_features_df) - test_size
    if train_size <= 0:
        raise ValueError("Not enough samples remaining for training after outlier removal and reserving test set.")
    
    # Log-transform skewed features
    skewed_features = ['total_bytes_at_production', 'total_vectors', 'total_parallelism']
    for feature in skewed_features:
        if feature in all_features_df.columns:
            all_features_df[f'log_{feature}'] = np.log1p(all_features_df[feature])
            all_features_df = all_features_df.drop(columns=[feature])
    
    all_features_df = all_features_df.fillna(0)
    
    constant_columns = [col for col in all_features_df.columns 
                       if col != 'execution_time' and all_features_df[col].nunique() == 1]
    all_features_df = all_features_df.drop(columns=constant_columns)
    print(f"Dropped {len(constant_columns)} constant columns")
    
    # Remove highly correlated features
    corr_matrix = all_features_df.drop(['execution_time'], axis=1).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    all_features_df = all_features_df.drop(columns=to_drop)
    print(f"Dropped {len(to_drop)} highly correlated features")
    
    if 'execution_time' in all_features_df.columns:
        all_features_df['execution_time_log'] = np.log1p(all_features_df['execution_time'])
    
    # Check if log-transformed columns exist before creating log_bytes_per_vector
    if 'log_total_bytes_at_production' in all_features_df.columns and 'log_total_vectors' in all_features_df.columns:
        all_features_df['log_bytes_per_vector'] = all_features_df['log_total_bytes_at_production'] / (all_features_df['log_total_vectors'] + 1e-8)
    else:
        print("Warning: 'log_total_bytes_at_production' or 'log_total_vectors' not found in DataFrame, skipping log_bytes_per_vector calculation")
        all_features_df['log_bytes_per_vector'] = 0.0  # Default value if calculation cannot be performed
    
    numeric_cols = all_features_df.select_dtypes(include=['number']).columns
    all_features_df = all_features_df[numeric_cols]
    
    # Split into train and test based on the adjusted sizes
    train_df = all_features_df.iloc[:train_size]
    test_df = all_features_df.iloc[train_size:]
    
    # Verify that test_df has the expected number of samples
    if len(test_df) != test_size:
        raise ValueError(f"Test set has {len(test_df)} samples, but expected {test_size}.")
    
    return train_df, test_df

def prepare_data_for_model(train_features, test_features, test_size=50):
    train_df, test_df = clean_and_transform_features(train_features, test_features, test_size=test_size)
    
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
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(test_df)
    y_test_scaled = scaler_y.transform(y_test)
    
    print(f"First 5 y_train scaled: {y_train_scaled[:5].flatten()}")
    print(f"First 5 y_test scaled: {y_test_scaled[:5].flatten()}")
    
    # Add noise for data augmentation
    noise = np.random.normal(0, 0.01, X_train_scaled.shape)
    X_train_scaled += noise
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    print(f"Input feature dimension: {X_train_scaled.shape[1]}")
    
    return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
            scaler_y, X_train_scaled.shape[1], is_log_transformed)

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / (self.hidden_size ** 0.5)
        self.v.data.uniform_(-stdv, stdv)
    
    def forward(self, hidden, encoder_outputs):
        batch_size, seq_len, _ = encoder_outputs.size()
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy @ self.v
        attention_weights = torch.softmax(energy, dim=1)
        context = (attention_weights.unsqueeze(-1) * encoder_outputs).sum(dim=1)
        return context

class ImprovedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=1, dropout_rate=0.3):
        super(ImprovedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 3
        
        # LSTM layers with batch_first=True
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=self.num_layers, batch_first=True, dropout=dropout_rate)
        
        # Attention layer
        self.attention = Attention(hidden_size)
        
        # Fully connected layers with batch normalization and residual connections
        self.fc1 = nn.Linear(hidden_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, output_size)
        
        # Residual connection
        self.residual = nn.Linear(hidden_size, 32)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        # Input shape: [batch_size, seq_len=1, feature_dim]
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Apply attention
        context = self.attention(hn[-1], lstm_out)
        
        # Fully connected layers
        x = self.fc1(context)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # Add residual connection
        residual = self.residual(context)
        x = x + residual
        
        x = self.fc4(x)
        return x

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=16):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

class CombinedLoss(nn.Module):
    def __init__(self, mape_weight=0.7, mse_weight=0.3, epsilon=1e-2):
        super(CombinedLoss, self).__init__()
        self.mape_weight = mape_weight
        self.mse_weight = mse_weight
        self.epsilon = epsilon
        self.mse_loss = nn.MSELoss()
    
    def forward(self, outputs, targets):
        mape = torch.mean(torch.abs((targets - outputs) / (targets + self.epsilon))) * 100
        mse = self.mse_loss(outputs, targets)
        return self.mape_weight * mape + self.mse_weight * mse

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=500, patience=30, checkpoint_path='checkpoint_lstm.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # Warm-up phase: linearly increase LR for the first 10 epochs
    warmup_epochs = 10
    base_lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr / warmup_epochs
    
    # Load checkpoint if it exists
    start_epoch, train_losses, val_losses, best_val_loss, epochs_no_improve = load_checkpoint(
        model, optimizer, scheduler, checkpoint_path
    )
    
    for epoch in range(start_epoch, num_epochs):
        # Warm-up phase
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print(f"Warm-up Epoch {epoch+1}, Learning Rate: {lr}")
        
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
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
        
        if epoch >= warmup_epochs:
            scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save checkpoint after each epoch
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
    
    # Load the best model state
    if os.path.exists('best_lstm_model.pth'):
        model.load_state_dict(torch.load('best_lstm_model.pth'))
        print("Loaded best model state from 'best_lstm_model.pth'")
    
    return train_losses, val_losses

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
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
        return 0, [], [], float('inf'), 0

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
    
    # Clip predictions to avoid negative values
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
        mdape = np.median(np.abs((y_test_actual[mask] - y_pred_actual[mask]) / y_test_actual[mask])) * 100
    else:
        mape = 0.0
        mdape = 0.0
    
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
    
    X_train, y_train, X_test, y_test, y_scaler, input_size, is_log_transformed = prepare_data_for_model(train_features, test_features, test_size=50)
    
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=16)
    
    model = ImprovedLSTMModel(
        input_size=input_size,
        hidden_size=256,
        output_size=1,
        dropout_rate=0.3
    )
    
    criterion = CombinedLoss(mape_weight=0.7, mse_weight=0.3, epsilon=1e-2)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    print("Building and training Improved LSTM model...")
    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        test_loader, 
        criterion, 
        optimizer, 
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
    plt.savefig('loss_improved_model.png')
    plt.close()
    print("Training plot saved as 'loss_improved_model.png'")
    
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
