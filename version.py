import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random

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

def save_scaler_params(scaler_X, scaler_y, is_log_transformed):
    scaler_X_data = {
        "feature_names": list(scaler_X.feature_names_in_),
        "means": scaler_X.center_.tolist(),
        "scales": scaler_X.scale_.tolist()
    }
    with open("scaler_X.json", "w") as f:
        json.dump(scaler_X_data, f)

    scaler_y_data = {
        "mean": float(scaler_y.center_[0]),
        "scale": float(scaler_y.scale_[0]),
        "is_log_transformed": is_log_transformed
    }
    with open("scaler_y.json", "w") as f:
        json.dump(scaler_y_data, f)

def extract_features_from_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
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
                        op_name, op_count = parts[0].strip(), int(parts[1].strip())
                        node_feature[f'op_{op_name.lower()}'] = op_count
            nodes_features.append(node_feature)
    
    if 'Edges' in programming_details:
        for edge in programming_details['Edges']:
            edges_features.append({
                'From': edge.get('From', ''),
                'To': edge.get('To', ''),
                'Name': edge.get('Name', '')
            })
    
    scheduling_features = data.get("scheduling_data", programming_details.get('Schedules', []))
    sched_features = []
    for sched in scheduling_features:
        sched_feature = {'Name': sched.get('Name', '')}
        if 'Details' in sched and 'scheduling_feature' in sched['Details']:
            sched_feature.update(sched['Details']['scheduling_feature'])
        sched_features.append(sched_feature)
    
    features = {
        'execution_time': min(max(execution_time, 1e-3), 1e6),  # Clip outliers
        'nodes_count': len(nodes_features),
        'edges_count': len(edges_features),
        'scheduling_count': len(sched_features),
        'node_edge_ratio': len(nodes_features) / len(edges_features) if len(edges_features) > 0 else 0,
        'log_nodes_count': np.log1p(len(nodes_features)),
        'log_edges_count': np.log1p(len(edges_features))
    }
    
    op_counts = {}
    for node in nodes_features:
        for key, value in node.items():
            if key.startswith('op_'):
                op_counts[key] = op_counts.get(key, 0) + value
    features.update(op_counts)
    
    if sched_features:
        important_metrics = [
            'bytes_at_production', 'bytes_at_realization', 'bytes_at_root', 'bytes_at_task',
            'inner_parallelism', 'outer_parallelism', 'num_productions', 'num_realizations',
            'num_scalars', 'num_vectors', 'points_computed_total', 'working_set'
        ]
        for metric in important_metrics:
            if metric in sched_features[0]:
                features[f'sched_{metric}'] = sched_features[0][metric]
                features[f'log_sched_{metric}'] = np.log1p(sched_features[0][metric])
        
        features['total_bytes_at_production'] = sum(sf.get('bytes_at_production', 0) for sf in sched_features)
        features['total_vectors'] = sum(sf.get('num_vectors', 0) for sf in sched_features)
        features['total_parallelism'] = sum(sf.get('inner_parallelism', 0) * sf.get('outer_parallelism', 1) for sf in sched_features)
        features['bytes_per_vector'] = features['total_bytes_at_production'] / (features['total_vectors'] + 1e-8)
        features['memory_pressure'] = sched_features[0]['working_set'] / sched_features[0]['bytes_at_production'] if sched_features[0].get('bytes_at_production', 0) > 0 else 0
    
    if nodes_features:
        features['avg_ops_per_node'] = sum(op_counts.values()) / len(nodes_features)
        features['op_diversity'] = len(op_counts) / len(nodes_features)
    
    return features

def process_directory(directory_path):
    all_features, file_names = [], []
    json_files = sorted(f for f in os.listdir(directory_path) if f.endswith('.json'))
    
    for filename in json_files:
        file_path = os.path.join(directory_path, filename)
        features = extract_features_from_file(file_path)
        if features:
            all_features.append(features)
            file_names.append(filename)
    
    return all_features, file_names

def process_main_directory(main_dir):
    all_features, all_file_names = [], []
    subdirs = sorted(d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d)))
    
    if not subdirs:
        raise ValueError(f"No subdirectories found in {main_dir}")
    
    for subdir in subdirs:
        subdir_path = os.path.join(main_dir, subdir)
        features, file_names = process_directory(subdir_path)
        if features:
            all_features.extend(features)
            all_file_names.extend(os.path.join(subdir, fname) for fname in file_names)
            print(f"Processed subdir {subdir}: {len(features)} files")
    
    if len(all_features) < 50:
        raise ValueError(f"Expected at least 50 files, found {len(all_features)}")
    
    combined = list(zip(all_features, all_file_names))
    random.shuffle(combined)
    all_features, all_file_names = zip(*combined)
    
    test_size = 50
    train_features, test_features = all_features[:-test_size], all_features[-test_size:]
    train_file_names, test_file_names = all_file_names[:-test_size], all_file_names[-test_size:]
    
    print(f"Total files: {len(all_features)}, Training: {len(train_features)}, Testing: {len(test_features)}")
    return train_features, test_features, list(test_file_names)

def clean_and_transform_features(train_features, test_features):
    df = pd.DataFrame(train_features + test_features).fillna(0)
    constant_cols = [col for col in df.columns if col != 'execution_time' and df[col].nunique() <= 1]
    df = df.drop(columns=constant_cols)
    print(f"Dropped {len(constant_cols)} constant columns")
    
    df['execution_time_log'] = np.log1p(df['execution_time'])
    if 'total_vectors' in df and df['total_vectors'].max() > 0:
        df['bytes_per_vector'] = df['total_bytes_at_production'] / (df['total_vectors'] + 1e-8)
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    df = df[numeric_cols]
    
    return df.iloc[:len(train_features)], df.iloc[len(train_features):]

def prepare_data_for_model(train_features, test_features):
    train_df, test_df = clean_and_transform_features(train_features, test_features)
    
    y_train = train_df['execution_time_log'].values.reshape(-1, 1)
    y_test = test_df['execution_time_log'].values.reshape(-1, 1)
    X_train = train_df.drop(['execution_time', 'execution_time_log'], axis=1)
    X_test = test_df.drop(['execution_time', 'execution_time_log'], axis=1)
    
    scaler_X, scaler_y = RobustScaler(), RobustScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    
    return (torch.FloatTensor(X_train_scaled).unsqueeze(1), torch.FloatTensor(y_train_scaled),
            torch.FloatTensor(X_test_scaled).unsqueeze(1), torch.FloatTensor(y_test_scaled),
            scaler_X, scaler_y, X_train_scaled.shape[1], True)

device = torch.device("cpu")

class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], output_size=1, dropout_rate=0.2):
        super().__init__()
        self.lstm_layers = nn.ModuleList([nn.LSTM(input_size if i == 0 else hidden_sizes[i-1], hs, batch_first=True, bidirectional=True)
                                         for i, hs in enumerate(hidden_sizes)])
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout_rate) for _ in hidden_sizes])
        self.attention = nn.MultiheadAttention(hidden_sizes[-1] * 2, num_heads=4, batch_first=True)  # Bidirectional doubles the size
        self.fc_layers = nn.ModuleList([nn.Linear(hidden_sizes[-1] * 2, hidden_sizes[-1]),
                                       nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2)])
        self.ln_layers = nn.ModuleList([nn.LayerNorm(hidden_sizes[-1]), nn.LayerNorm(hidden_sizes[-1] // 2)])
        self.output_layer = nn.Linear(hidden_sizes[-1] // 2, output_size)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.residual_adapter = nn.Linear(hidden_sizes[-1] * 2, hidden_sizes[-1] // 2) if hidden_sizes[-1] * 2 != hidden_sizes[-1] // 2 else None
    
    def forward(self, x):
        for i, (lstm, dropout) in enumerate(zip(self.lstm_layers, self.dropout_layers)):
            x, _ = lstm(x)
            if i < len(self.lstm_layers) - 1:
                x = dropout(x)
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.squeeze(1)  # Remove sequence dimension
        residual = x if not self.residual_adapter else self.residual_adapter(x)
        x = self.leaky_relu(self.ln_layers[0](self.fc_layers[0](x)))
        x = self.leaky_relu(self.ln_layers[1](self.fc_layers[1](x)))
        x = x + residual
        return self.output_layer(x)

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32):
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=200, patience=30):
    model.to(device)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, verbose=True)
    best_val_loss, epochs_no_improve, best_model_state = float('inf'), 0, None
    train_losses, val_losses = [], []
    
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
                val_loss += criterion(outputs, targets).item() * inputs.size(0)
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss - 0.001:  # Minimum delta for improvement
            best_val_loss, epochs_no_improve = val_loss, 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                model.load_state_dict(best_model_state)
                break
    
    return train_losses, val_losses

def evaluate_model(model, X_test, y_test, y_scaler, file_names_test, is_log_transformed=True):
    model.eval().to(device)
    with torch.no_grad():
        y_pred_scaled = model(X_test.to(device)).cpu().numpy()
    y_test = y_test.cpu().numpy()
    
    y_test_transformed = y_scaler.inverse_transform(y_test)
    y_pred_transformed = y_scaler.inverse_transform(y_pred_scaled)
    y_test_actual = np.expm1(y_test_transformed) if is_log_transformed else y_test_transformed
    y_pred_actual = np.expm1(y_pred_transformed) if is_log_transformed else y_pred_transformed
    
    results_by_subfolder = {}
    for i, file_path in enumerate(file_names_test):
        subfolder = file_path.split('/')[0]
        results_by_subfolder.setdefault(subfolder, []).append({
            'file': file_path,
            'actual': y_test_actual[i][0],
            'predicted': y_pred_actual[i][0],
            'error_percentage': abs(y_test_actual[i][0] - y_pred_actual[i][0]) / max(y_test_actual[i][0], 1e-3) * 100
        })
    
    for subfolder, results in results_by_subfolder.items():
        print(f"\nResults for {subfolder}:")
        for r in results:
            print(f"File: {r['file']}, Actual: {r['actual']:.2f} ms, Predicted: {r['predicted']:.2f} ms, Error: {r['error_percentage']:.2f}%")
    
    mse = float(np.mean((y_test_actual - y_pred_actual) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_test_actual - y_pred_actual)))
    mape = float(np.mean(np.abs((y_test_actual - y_pred_actual) / np.maximum(y_test_actual, 1e-3))) * 100)
    r2 = float(r2_score(y_test_actual, y_pred_actual))
    avg_actual = float(np.mean(y_test_actual))
    avg_predicted = float(np.mean(y_pred_actual))
    
    print(f"\nOverall Model Performance:\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nMAPE: {mape:.2f}%\nR²: {r2:.4f}\nAvg Actual: {avg_actual:.2f} ms\nAvg Predicted: {avg_predicted:.2f} ms")
    
    performance_metrics = {
        "mse": mse, "rmse": rmse, "mae": mae, "mape": mape, "r2": r2,
        "avg_actual_execution_time_ms": avg_actual,
        "avg_predicted_execution_time_ms": avg_predicted
    }
    with open("model_performance_metrics.json", "w") as f:
        json.dump(performance_metrics, f, indent=4)
    print("Performance metrics saved to 'model_performance_metrics.json'")
    
    return y_test_actual, y_pred_actual

def main(main_dir):
    print(f"Processing main directory: {main_dir}")
    train_features, test_features, test_file_names = process_main_directory(main_dir)
    if not train_features or not test_features:
        print("Error: No valid data found")
        return None
    
    X_train, y_train, X_test, y_test, scaler_X, y_scaler, input_size, is_log_transformed = prepare_data_for_model(train_features, test_features)
    save_scaler_params(scaler_X, y_scaler, is_log_transformed)
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test)
    
    model = EnhancedLSTMModel(input_size=input_size)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    
    print("Training Enhanced LSTM model...")
    train_losses, val_losses = train_model(model, train_loader, test_loader, criterion, optimizer)
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(model, X_test, y_test, y_scaler, test_file_names)
    
    print("\nSaving model as 'lstm_model.pt'...")
    model.eval().to(device)
    try:
        traced_model = torch.jit.trace(model, torch.randn(1, 1, input_size).to(device))
        traced_model.save("lstm_model.pt")
        print("Model saved successfully")
    except Exception as e:
        print(f"Error saving model: {str(e)}")
    
    return model, y_scaler, y_test_actual, y_pred_actual

if __name__ == "__main__":
    random.seed(42)
    main_dir = "synthetic_data"
    model, y_scaler, y_test_actual, y_pred_actual = main(main_dir)
