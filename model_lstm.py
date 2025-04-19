import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
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
            print(f"Warning: 'total_execution_time_ms' not found in {file_path}, using last value: {execution_time} ms")
            return float(execution_time)
        
        print(f"Error: No valid execution time found in {file_path}")
        return None
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def extract_features_from_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    execution_time = get_execution_time(file_path)
    if execution_time is None:
        return None
    
    execution_time = np.clip(execution_time, 1.0, 15000.0)  # Adjusted upper bound
    
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
                        op_name = parts[0].strip().lower()
                        op_count = int(parts[1].strip())
                        node_feature[f'op_{op_name}'] = op_count
            nodes_features.append(node_feature)
    
    if 'Edges' in programming_details:
        for edge in programming_details['Edges']:
            edges_features.append({
                'From': edge.get('From', ''),
                'To': edge.get('To', ''),
                'Name': edge.get('Name', '')
            })
    
    scheduling_features = data.get("scheduling_data", programming_details.get('Schedules', []))
    
    features = {
        'execution_time': execution_time,
        'nodes_count': len(nodes_features),
        'edges_count': len(edges_features),
        'scheduling_count': len(scheduling_features)
    }
    
    features['node_edge_ratio'] = len(nodes_features) / len(edges_features) if edges_features else 0
    
    op_counts = {}
    for node in nodes_features:
        for key, value in node.items():
            if key.startswith('op_'):
                op_counts[key] = op_counts.get(key, 0) + value
    features.update(op_counts)
    
    if scheduling_features:
        metrics = [
            'bytes_at_production', 'bytes_at_realization', 'inner_parallelism', 
            'outer_parallelism', 'num_vectors', 'working_set'
        ]
        if isinstance(scheduling_features[0], dict):
            for metric in metrics:
                features[f'sched_{metric}'] = scheduling_features[0].get(metric, 0)
        
        features['total_bytes_at_production'] = sum(sf.get('bytes_at_production', 0) for sf in scheduling_features if isinstance(sf, dict))
        features['total_vectors'] = sum(sf.get('num_vectors', 0) for sf in scheduling_features if isinstance(sf, dict))
        features['total_parallelism'] = sum(sf.get('inner_parallelism', 0) * sf.get('outer_parallelism', 1) for sf in scheduling_features if isinstance(sf, dict))
        
        features['bytes_per_vector'] = features['total_bytes_at_production'] / (features['total_vectors'] + 1e-8)
        if 'working_set' in scheduling_features[0] and scheduling_features[0].get('bytes_at_production', 0) > 0:
            features['memory_pressure'] = scheduling_features[0]['working_set'] / scheduling_features[0]['bytes_at_production']
    
    if nodes_features:
        features['avg_ops_per_node'] = sum(op_counts.values()) / len(nodes_features)
        features['op_diversity'] = len(op_counts) / len(nodes_features)
    
    return features

def process_directory(directory_path):
    all_features = []
    file_names = []
    
    json_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.json')])
    for filename in json_files:
        file_path = os.path.join(directory_path, filename)
        features = extract_features_from_file(file_path)
        if features:
            all_features.append(features)
            file_names.append(filename)
    
    return all_features, file_names

def process_main_directory(main_dir):
    all_features = []
    all_file_names = []
    
    subdirs = sorted([d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))])
    if not subdirs:
        raise ValueError(f"No subdirectories found in {main_dir}")
    
    for subdir in subdirs:
        subdir_path = os.path.join(main_dir, subdir)
        features, file_names = process_directory(subdir_path)
        if features:
            all_features.extend(features)
            all_file_names.extend([os.path.join(subdir, fname) for fname in file_names])
            print(f"Processed {subdir}: {len(features)} files")
    
    if len(all_features) < 50:
        raise ValueError(f"Expected at least 50 files, found {len(all_features)}")
    
    combined = list(zip(all_features, all_file_names))
    random.shuffle(combined)
    all_features, all_file_names = zip(*combined)
    
    test_size = 50
    train_features = all_features[:-test_size]
    test_features = all_features[-test_size:]
    train_file_names = all_file_names[:-test_size]
    test_file_names = all_file_names[-test_size:]
    
    print(f"Total files: {len(all_features)}, Training: {len(train_features)}, Testing: {len(test_features)}")
    return train_features, test_features, list(test_file_names)

def clean_and_transform_features(train_features, test_features):
    all_features_df = pd.DataFrame(train_features + test_features).fillna(0)
    
    constant_columns = [col for col in all_features_df.columns if col != 'execution_time' and all_features_df[col].nunique() == 1]
    all_features_df = all_features_df.drop(columns=constant_columns)
    print(f"Dropped {len(constant_columns)} constant columns")
    
    corr_matrix = all_features_df.drop(['execution_time'], axis=1).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    all_features_df = all_features_df.drop(columns=to_drop)
    print(f"Dropped {len(to_drop)} highly correlated features")
    
    all_features_df['execution_time_log'] = np.log1p(all_features_df['execution_time'])
    
    # Add polynomial features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    numeric_cols = all_features_df.select_dtypes(include=['number']).columns.drop(['execution_time', 'execution_time_log'])
    poly_features = poly.fit_transform(all_features_df[numeric_cols])
    poly_feature_names = poly.get_feature_names_out(numeric_cols)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
    all_features_df = pd.concat([all_features_df, poly_df], axis=1)
    
    train_size = len(train_features)
    train_df = all_features_df.iloc[:train_size]
    test_df = all_features_df.iloc[train_size:]
    
    return train_df, test_df

def prepare_data_for_model(train_features, test_features):
    train_df, test_df = clean_and_transform_features(train_features, test_features)
    
    y_train = train_df['execution_time_log'].values.reshape(-1, 1)
    y_test = test_df['execution_time_log'].values.reshape(-1, 1)
    X_train = train_df.drop(['execution_time', 'execution_time_log'], axis=1)
    X_test = test_df.drop(['execution_time', 'execution_time_log'], axis=1)
    
    scaler_X = StandardScaler()
    scaler_y = RobustScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    print(f"Input feature dimension: {X_train_scaled.shape[1]}")
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler_y, X_train_scaled.shape[1], True

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / (hidden_size ** 0.5)
        self.v.data.uniform_(-stdv, stdv)
    
    def forward(self, hidden, lstm_output):
        batch_size = lstm_output.size(0)
        seq_len = lstm_output.size(1)
        
        hidden = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)  # [batch, seq_len, hidden_size]
        energy = torch.tanh(self.attn(torch.cat((hidden, lstm_output), dim=2)))
        energy = energy.transpose(1, 2)  # [batch, hidden_size, seq_len]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)  # [batch, 1, hidden_size]
        attention = torch.bmm(v, energy).squeeze(1)  # [batch, seq_len]
        return torch.softmax(attention, dim=1)

class PerfectLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=512, output_size=1, dropout_rate=0.3):
        super(PerfectLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=3, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size)
        
        self.fc1 = nn.Linear(hidden_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, output_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        lstm_out, (hn, _) = self.lstm(x)
        lstm_out = lstm_out[:, :, :self.hidden_size] + lstm_out[:, :, self.hidden_size:]  # Sum bidirectional outputs
        
        attn_weights = self.attention(hn, lstm_out)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        
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
        
        x = self.fc4(x)
        return x

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=8):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

class CustomLoss(nn.Module):
    def __init__(self, mse_weight=0.6, mape_weight=0.4, high_error_penalty=0.1, epsilon=1e-2):
        super(CustomLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.mse_weight = mse_weight
        self.mape_weight = mape_weight
        self.high_error_penalty = high_error_penalty
        self.epsilon = epsilon
    
    def forward(self, outputs, targets):
        mse_loss = self.mse(outputs, targets)
        mape_loss = torch.mean(torch.abs((targets - outputs) / (targets + self.epsilon))) * 100
        
        # Penalize large errors on high-value targets
        high_error = torch.mean(torch.where(targets > 1.0, torch.abs(targets - outputs), torch.zeros_like(targets))) * self.high_error_penalty
        
        return self.mse_weight * mse_loss + self.mape_weight * mape_loss + high_error

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=400, patience=60):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=20, verbose=True)
    
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
    
    if best_model_state and epochs_no_improve > 0:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses

def evaluate_model(model, X_test, y_test, y_scaler, file_names_test, is_log_transformed=True, original_execution_times=None):
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
    
    y_test_actual = np.expm1(y_test_transformed) if is_log_transformed else y_test_transformed
    y_pred_actual = np.expm1(y_pred_transformed) if is_log_transformed else y_pred_transformed
    y_pred_actual = np.maximum(y_pred_actual, 1e-2)  # Avoid negative predictions
    
    results_by_subfolder = {}
    for i, file_path in enumerate(file_names_test):
        subfolder = file_path.split('/')[0]
        if subfolder not in results_by_subfolder:
            results_by_subfolder[subfolder] = []
        
        actual_val = y_test_actual[i][0]
        pred_val = y_pred_actual[i][0]
        error_percentage = min(abs(actual_val - pred_val) / actual_val * 100 if actual_val > 0 else 0, 1000.0)
        
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
    mape = np.mean(np.abs((y_test_actual[mask] - y_pred_actual[mask]) / y_test_actual[mask])) * 100 if mask.sum() > 0 else 0.0
    
    print("\nOverall Model Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    return y_test_actual, y_pred_actual

def main(main_dir):
    print(f"Processing main directory: {main_dir}")
    train_features, test_features, test_file_names = process_main_directory(main_dir)
    
    if not train_features or not test_features:
        print("Error: No valid data found")
        return None
    
    original_execution_times = {fname: feat['execution_time'] for feat, fname in zip(test_features, test_file_names)}
    
    X_train, y_train, X_test, y_test, y_scaler, input_size, is_log_transformed = prepare_data_for_model(train_features, test_features)
    
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=8)
    
    model = PerfectLSTMModel(input_size=input_size, hidden_size=512, output_size=1, dropout_rate=0.3)
    criterion = CustomLoss(mse_weight=0.6, mape_weight=0.4, high_error_penalty=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
    
    print("Training Perfect LSTM model...")
    train_losses, val_losses = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=400, patience=60)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_perfect_model_v2.png')
    plt.close()
    print("Training plot saved as 'loss_perfect_model_v2.png'")
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(model, X_test, y_test, y_scaler, test_file_names, is_log_transformed, original_execution_times)
    
    print("\nSaving model as 'lstm_model_v2.pt'...")
    device = next(model.parameters()).device
    try:
        sample_input = torch.randn(1, 1, input_size).to(device)
        traced_model = torch.jit.trace(model, sample_input)
        traced_model.save("lstm_model_v2.pt")
        print("Model saved as 'lstm_model_v2.pt'")
    except Exception as e:
        print(f"Error saving model: {str(e)}")
    
    return model, y_scaler, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "synthetic_data"
    random.seed(42)
    result = main(main_dir)
    if result:
        model, y_scaler, y_test_actual, y_pred_actual = result
