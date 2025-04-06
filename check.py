import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
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
                if execution_time is not None and execution_time > 0:  # Ensure positive execution time
                    return float(execution_time)
        
        print(f"Warning: 'total_execution_time_ms' not found in 'Schedules' of {file_path}")
        last_value = schedules[-1]["value"]
        return float(last_value) if last_value > 0 else None
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def save_scaler_params(scaler_X, scaler_y, is_log_transformed):
    scaler_X_data = {
        "feature_names": list(scaler_X.feature_names_in_),
        "means": scaler_X.mean_.tolist(),
        "scales": scaler_X.scale_.tolist()
    }
    with open("scaler_X.json", "w") as f:
        json.dump(scaler_X_data, f)

    scaler_y_data = {
        "mean": float(scaler_y.mean_[0]),
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
        print(f"Warning: Invalid execution time in {file_path}")
        return None
    
    nodes_features = []
    edges_features = []
    programming_details = data.get("programming_details")
    
    if programming_details:
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
    scheduling_data = data.get("scheduling_data")
    if not scheduling_data and programming_details and 'Schedules' in programming_details:
        scheduling_data = programming_details['Schedules']
    
    if scheduling_data:
        for sched in scheduling_data:
            sched_feature = {'Name': sched.get('Name', '')}
            if 'Details' in sched and 'scheduling_feature' in sched['Details']:
                sf = sched['Details']['scheduling_feature']
                sched_feature.update(sf)
            scheduling_features.append(sched_feature)
    
    features = {
        'execution_time': execution_time,
        'nodes_count': len(nodes_features),
        'edges_count': len(edges_features),
        'scheduling_count': len(scheduling_features)
    }
    
    features['node_edge_ratio'] = len(nodes_features) / max(len(edges_features), 1)
    
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
        for metric in important_metrics:
            features[f'sched_{metric}'] = sum(sf.get(metric, 0) for sf in scheduling_features) / len(scheduling_features) if scheduling_features else 0
        
        total_bytes_at_production = sum(sf.get('bytes_at_production', 0) for sf in scheduling_features)
        total_vectors = sum(sf.get('num_vectors', 0) for sf in scheduling_features)
        total_points = sum(sf.get('points_computed_total', 0) for sf in scheduling_features)
        total_parallelism = sum(sf.get('inner_parallelism', 0) * sf.get('outer_parallelism', 1) for sf in scheduling_features)
        
        features['total_bytes_at_production'] = total_bytes_at_production
        features['total_vectors'] = total_vectors
        features['total_parallelism'] = total_parallelism
        features['total_points_computed'] = total_points
        
        features['bytes_per_vector'] = total_bytes_at_production / max(total_vectors, 1e-8)
        features['points_per_vector'] = total_points / max(total_vectors, 1e-8)
        features['bytes_per_point'] = total_bytes_at_production / max(total_points, 1e-8)
    
    if nodes_features:
        total_ops = sum(op_counts.values())
        features['avg_ops_per_node'] = total_ops / len(nodes_features)
        features['op_diversity'] = len(op_counts) / len(nodes_features)
        features['ops_per_byte'] = total_ops / max(features['total_bytes_at_production'], 1e-8)
    
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
                        if col != 'execution_time' and all_features_df[col].nunique() <= 1]
    all_features_df = all_features_df.drop(columns=constant_columns)
    print(f"Dropped {len(constant_columns)} constant columns")
    
    # Clip outliers before log transformation
    all_features_df['execution_time'] = np.clip(all_features_df['execution_time'], 
                                                all_features_df['execution_time'].quantile(0.01), 
                                                all_features_df['execution_time'].quantile(0.99))
    all_features_df['execution_time_log'] = np.log1p(all_features_df['execution_time'])
    
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
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(train_df)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(test_df)
    y_test_scaled = scaler_y.transform(y_test)
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    print(f"Input feature dimension: {X_train_scaled.shape[1]}")
    
    return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
            scaler_X, scaler_y, X_train_scaled.shape[1], True)

class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], output_size=1, dropout_rate=0.2, num_heads=4):
        super(EnhancedLSTMModel, self).__init__()
        
        # Bidirectional LSTM layers
        self.lstm_layers = nn.ModuleList()
        self.lstm_layers.append(nn.LSTM(input_size, hidden_sizes[0], batch_first=True, bidirectional=True))
        for i in range(1, len(hidden_sizes)):
            self.lstm_layers.append(nn.LSTM(hidden_sizes[i-1] * 2, hidden_sizes[i], batch_first=True, bidirectional=True))
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_sizes[-1] * 2, num_heads, dropout=dropout_rate, batch_first=True)
        
        # Fully connected layers with residual connections
        self.fc1 = nn.Linear(hidden_sizes[-1] * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.output_layer = nn.Linear(64, output_size)
        
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.residual_proj = nn.Linear(hidden_sizes[-1] * 2, 64) if hidden_sizes[-1] * 2 != 64 else None
    
    def forward(self, x):
        lstm_out = x
        for i, lstm in enumerate(self.lstm_layers):
            lstm_out, _ = lstm(lstm_out)
            if i < len(self.lstm_layers) - 1:
                lstm_out = self.dropout(lstm_out)
        
        # Multi-head attention (sequence length is 1, so we process directly)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        context = attn_out.squeeze(1)  # Remove sequence dimension
        
        x = self.fc1(context)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        
        # Residual connection
        residual = context if self.residual_proj is None else self.residual_proj(context)
        x = x + residual
        x = self.dropout(x)
        
        output = self.output_layer(x)
        return output

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=200, patience=30):
    device = torch.device('cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
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
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss at epoch {epoch+1}")
                return None, None
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
        
        if val_loss < best_val_loss and not np.isnan(val_loss) and not np.isinf(val_loss):
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            model.load_state_dict(best_model_state)
            break
    
    if best_model_state is not None and epochs_no_improve > 0:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses

def evaluate_model(model, X_test, y_test, y_scaler, file_names_test, is_log_transformed=True):
    device = torch.device('cpu')
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
    
    avg_actual = np.mean(y_test_actual)
    avg_predicted = np.mean(y_pred_actual)
    
    results_by_subfolder = {}
    for i, file_path in enumerate(file_names_test):
        subfolder = file_path.split('/')[0]
        if subfolder not in results_by_subfolder:
            results_by_subfolder[subfolder] = []
        
        pred = max(y_pred_actual[i][0], 0)
        results_by_subfolder[subfolder].append({
            'file': file_path,
            'actual': y_test_actual[i][0],
            'predicted': pred,
            'error_percentage': abs(y_test_actual[i][0] - pred) / y_test_actual[i][0] * 100 if y_test_actual[i][0] > 0 else 0
        })
    
    for subfolder, results in results_by_subfolder.items():
        print(f"\nResults for {subfolder}:")
        for result in results:
            print(f"File: {result['file']}")
            print(f"  Actual execution time: {result['actual']:.2f} ms")
            print(f"  Predicted execution time: {result['predicted']:.2f} ms")
            print(f"  Error percentage: {result['error_percentage']:.2f}%")
    
    mse = np.mean((y_test_actual - y_pred_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_actual - y_pred_actual))
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / (y_test_actual + 1e-8))) * 100
    
    print("\nOverall Model Performance:")
    print(f"Average Actual Execution Time: {avg_actual:.2f} ms")
    print(f"Average Predicted Execution Time: {avg_predicted:.2f} ms")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    return y_test_actual, y_pred_actual, avg_actual, avg_predicted

def main(main_dir):
    print(f"Processing main directory: {main_dir}")
    train_features, test_features, test_file_names = process_main_directory(main_dir)
    
    print(f"Total training samples: {len(train_features)} (randomly selected)")
    print(f"Total test samples: {len(test_features)} (50 randomly selected)")
    
    if len(train_features) == 0 or len(test_features) == 0:
        print("Error: No valid training or test data found")
        return None
    
    X_train, y_train, X_test, y_test, scaler_X, y_scaler, input_size, is_log_transformed = prepare_data_for_model(train_features, test_features)
    save_scaler_params(scaler_X, y_scaler, is_log_transformed)
    
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32)
    
    model = EnhancedLSTMModel(
        input_size=input_size,
        hidden_sizes=[256, 128, 64],
        output_size=1,
        dropout_rate=0.2,
        num_heads=4
    )
    
    criterion = nn.HuberLoss(delta=0.5)  # Smaller delta for stricter loss
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    print("Building and training Enhanced LSTM model...")
    train_losses, val_losses = train_model(
        model, train_loader, test_loader,
        criterion, optimizer,
        num_epochs=200, patience=30
    )
    
    if train_losses is None or val_losses is None:
        print("Training failed due to invalid loss values")
        return None
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual, avg_actual, avg_predicted = evaluate_model(
        model, X_test, y_test, y_scaler, test_file_names, is_log_transformed
    )
    
    print("\nSaving the trained model as 'lstm_model.pt'...")
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    print(f"Model is on device: {device}")
    
    try:
        sample_input = torch.randn(1, 1, input_size).to(device)
        traced_model = torch.jit.trace(model, sample_input)
        traced_model.save("lstm_model.pt")
        print("Model successfully saved as 'lstm_model.pt'")
    except Exception as e:
        print(f"Error saving the model: {str(e)}")
    
    print(f"\nSummary for Comparison:")
    print(f"Model: EnhancedLSTM")
    print(f"Average Actual Execution Time: {avg_actual:.2f} ms")
    print(f"Average Predicted Execution Time: {avg_predicted:.2f} ms")
    
    return model, y_scaler, y_test_actual, y_pred_actual, avg_actual, avg_predicted

if __name__ == "__main__":
    main_dir = "synthetic_data"
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    model, y_scaler, y_test_actual, y_pred_actual, avg_actual, avg_predicted = main(main_dir)
