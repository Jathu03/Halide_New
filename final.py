import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    programming_details = None
    for key, value in data.items():
        if key == "programming_details":
            programming_details = value
            break
    
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
    scheduling_data = None
    for key, value in data.items():
        if key == "scheduling_data":
            scheduling_data = value
            break
    
    if not scheduling_data and 'Schedules' in programming_details:
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
    
    # Add node to edge ratio feature
    if len(nodes_features) > 0 and len(edges_features) > 0:
        features['node_edge_ratio'] = len(nodes_features) / len(edges_features)
    else:
        features['node_edge_ratio'] = 0
    
    # Parse op counts with improved handling
    op_counts = {}
    for node in nodes_features:
        for key, value in node.items():
            if key.startswith('op_'):
                op_counts[key] = op_counts.get(key, 0) + value
    features.update(op_counts)
    
    # Extract more detailed scheduling metrics
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
        
        # Calculate aggregated metrics
        total_bytes_at_production = sum(sf.get('bytes_at_production', 0) for sf in scheduling_features if isinstance(sf, dict))
        total_vectors = sum(sf.get('num_vectors', 0) for sf in scheduling_features if isinstance(sf, dict))
        total_parallelism = sum(sf.get('inner_parallelism', 0) * sf.get('outer_parallelism', 1) for sf in scheduling_features if isinstance(sf, dict))
        
        features['total_bytes_at_production'] = total_bytes_at_production
        features['total_vectors'] = total_vectors
        features['total_parallelism'] = total_parallelism
        
        # Add derived features
        if total_vectors > 0:
            features['bytes_per_vector'] = total_bytes_at_production / total_vectors if total_vectors > 0 else 0
        
        # Add memory pressure indicator
        if 'working_set' in scheduling_features[0] and 'bytes_at_production' in scheduling_features[0]:
            features['memory_pressure'] = scheduling_features[0]['working_set'] / scheduling_features[0]['bytes_at_production'] if scheduling_features[0]['bytes_at_production'] > 0 else 0
    
    # Add node complexity indicators
    if len(nodes_features) > 0:
        op_types = sum(1 for k in op_counts.keys())
        features['avg_ops_per_node'] = sum(op_counts.values()) / len(nodes_features)
        features['op_diversity'] = op_types / len(nodes_features) if len(nodes_features) > 0 else 0
    
    return features

def process_directory(directory_path):
    """Process all JSON files in a directory and split into train and test sets."""
    all_features = []
    file_names = []
    
    # Get all JSON files in the directory
    json_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.json')])
    
    if len(json_files) < 32:
        print(f"Warning: Expected at least 32 files in {directory_path}, found {len(json_files)}")
    
    # Process each file and extract features
    for filename in json_files:
        file_path = os.path.join(directory_path, filename)
        features = extract_features_from_file(file_path)
        if features is not None:
            all_features.append(features)
            file_names.append(filename)
    
    # Ensure we have at least 32 files to work with
    if len(all_features) < 32:
        print(f"Warning: Only {len(all_features)} valid files found in {directory_path}")
        return None, None, None
    
    # Split into training (first 30) and testing (last 2)
    train_features = all_features[:30]
    test_features = all_features[30:32]
    test_file_names = file_names[30:32]
    
    return train_features, test_features, test_file_names

def process_main_directory(main_dir):
    """Process all subdirectories, splitting each into train/test sets."""
    all_train_features = []
    all_test_features = []
    all_test_file_names = []
    
    # Get all subdirectories
    subdirs = sorted([d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))])
    
    if len(subdirs) < 1:
        raise ValueError(f"Expected at least 1 subdirectory in {main_dir}, found {len(subdirs)}")
    
    # Process each subdirectory
    for subdir in subdirs:
        subdir_path = os.path.join(main_dir, subdir)
        train_features, test_features, test_file_names = process_directory(subdir_path)
        
        if train_features is None or test_features is None:
            print(f"Skipping {subdir} due to insufficient data")
            continue
        
        all_train_features.extend(train_features)
        all_test_features.extend(test_features)
        all_test_file_names.extend([os.path.join(subdir, fname) for fname in test_file_names])
        
        print(f"Processed subdir {subdir}: {len(train_features)} training files, {len(test_features)} test files")
    
    return all_train_features, all_test_features, all_test_file_names

def clean_and_transform_features(train_features, test_features):
    """Clean, transform, and augment features for improved model performance"""
    all_features_df = pd.DataFrame(train_features + test_features)
    
    # Handle missing values
    all_features_df = all_features_df.fillna(0)
    
    # Drop columns with constant values
    constant_columns = [col for col in all_features_df.columns 
                       if col != 'execution_time' and all_features_df[col].nunique() == 1]
    all_features_df = all_features_df.drop(columns=constant_columns)
    print(f"Dropped {len(constant_columns)} constant columns")
    
    # Log transform the target variable to handle skewness
    if 'execution_time' in all_features_df.columns:
        all_features_df['execution_time_log'] = np.log1p(all_features_df['execution_time'])
    
    # Create ratio features for potentially important metrics
    if 'total_vectors' in all_features_df.columns and all_features_df['total_vectors'].max() > 0:
        all_features_df['bytes_per_vector'] = all_features_df['total_bytes_at_production'] / (all_features_df['total_vectors'] + 1e-8)
    
    # Feature selection - keep only numeric columns
    numeric_cols = all_features_df.select_dtypes(include=['number']).columns
    all_features_df = all_features_df[numeric_cols]
    
    # Split back into train and test
    train_size = len(train_features)
    train_df = all_features_df.iloc[:train_size]
    test_df = all_features_df.iloc[train_size:]
    
    return train_df, test_df

def prepare_data_for_model(train_features, test_features):
    """Prepare the training and testing data for the model."""
    # Clean and transform features
    train_df, test_df = clean_and_transform_features(train_features, test_features)
    
    # Use log-transformed target if available
    if 'execution_time_log' in train_df.columns:
        y_train = train_df['execution_time_log'].values.reshape(-1, 1)
        y_test = test_df['execution_time_log'].values.reshape(-1, 1)
        train_df = train_df.drop(['execution_time', 'execution_time_log'], axis=1)
        test_df = test_df.drop(['execution_time', 'execution_time_log'], axis=1)
    else:
        y_train = train_df['execution_time'].values.reshape(-1, 1)
        y_test = test_df['execution_time'].values.reshape(-1, 1)
        train_df = train_df.drop('execution_time', axis=1)
        test_df = test_df.drop('execution_time', axis=1)
    
    # Normalize features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(train_df)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(test_df)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Create tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    print(f"Input feature dimension: {X_train_scaled.shape[1]}")
    
    return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
            scaler_y, X_train_scaled.shape[1], 'execution_time_log' in train_df.columns)

class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=1, dropout_rate=0.3):
        super(EnhancedLSTMModel, self).__init__()
        
        # LSTM layers with increasing complexity
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # First LSTM layer
        self.lstm_layers.append(nn.LSTM(input_size, hidden_sizes[0], batch_first=True))
        self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        # Additional LSTM layers
        for i in range(1, len(hidden_sizes)):
            self.lstm_layers.append(nn.LSTM(hidden_sizes[i-1], hidden_sizes[i], batch_first=True))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_sizes[-1], 1)
        
        # Fully connected layers with batch normalization
        self.fc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        # First FC layer after LSTM
        self.fc_layers.append(nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2))
        self.bn_layers.append(nn.BatchNorm1d(hidden_sizes[-1] // 2))
        
        # Second FC layer
        self.fc_layers.append(nn.Linear(hidden_sizes[-1] // 2, hidden_sizes[-1] // 4))
        self.bn_layers.append(nn.BatchNorm1d(hidden_sizes[-1] // 4))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1] // 4, output_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
        # Residual connections
        self.has_residual = (hidden_sizes[-1] // 4 == hidden_sizes[-1] // 2)
        if not self.has_residual:
            self.residual_adapter = nn.Linear(hidden_sizes[-1] // 2, hidden_sizes[-1] // 4)
        
    def attention_net(self, lstm_output):
        attn_weights = self.attention(lstm_output).squeeze(2)
        soft_attn_weights = torch.softmax(attn_weights, 1)
        context = torch.bmm(soft_attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context
        
    def forward(self, x):
        # Process through LSTM layers
        lstm_out = x
        for i, (lstm, dropout) in enumerate(zip(self.lstm_layers, self.dropout_layers)):
            lstm_out, _ = lstm(lstm_out)
            if i < len(self.lstm_layers) - 1:  # Apply dropout except for the last layer
                lstm_out = dropout(lstm_out)
        
        # Apply attention
        attn_output = self.attention_net(lstm_out)
        
        # First FC layer
        fc_out = self.fc_layers[0](attn_output)
        fc_out = self.bn_layers[0](fc_out)
        fc_out = self.leaky_relu(fc_out)
        
        # Save for residual connection
        residual = fc_out
        if not self.has_residual:
            residual = self.residual_adapter(residual)
        
        # Second FC layer
        fc_out = self.fc_layers[1](fc_out)
        fc_out = self.bn_layers[1](fc_out)
        fc_out = self.leaky_relu(fc_out)
        
        # Add residual connection
        fc_out = fc_out + residual
        
        # Output layer
        output = self.output_layer(fc_out)
        
        return output

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=150, patience=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
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
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early stopping check
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
    
    if best_model_state is not None and epochs_no_improve > 0:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses

def evaluate_model(model, X_test, y_test, y_scaler, file_names_test, is_log_transformed=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    X_test = X_test.to(device)
    with torch.no_grad():
        y_pred_scaled = model(X_test)
    
    y_pred_scaled = y_pred_scaled.cpu().numpy()
    y_test = y_test.cpu().numpy()
    
    # Inverse transform the predictions
    y_test_transformed = y_scaler.inverse_transform(y_test)
    y_pred_transformed = y_scaler.inverse_transform(y_pred_scaled)
    
    # If log-transformed, convert back to original scale
    if is_log_transformed:
        y_test_actual = np.expm1(y_test_transformed)
        y_pred_actual = np.expm1(y_pred_transformed)
    else:
        y_test_actual = y_test_transformed
        y_pred_actual = y_pred_transformed
    
    # Group results by subfolder
    results_by_subfolder = {}
    for i, file_path in enumerate(file_names_test):
        subfolder = file_path.split('/')[0]
        if subfolder not in results_by_subfolder:
            results_by_subfolder[subfolder] = []
        
        results_by_subfolder[subfolder].append({
            'file': file_path,
            'actual': y_test_actual[i][0],
            'predicted': y_pred_actual[i][0],
            'error_percentage': abs(y_test_actual[i][0] - y_pred_actual[i][0]) / y_test_actual[i][0] * 100 if y_test_actual[i][0] > 0 else 0
        })
    
    # Print results for each subfolder
    for subfolder, results in results_by_subfolder.items():
        print(f"\nResults for {subfolder}:")
        for result in results:
            print(f"File: {result['file']}")
            print(f"  Actual execution time: {result['actual']:.2f} ms")
            print(f"  Predicted execution time: {result['predicted']:.2f} ms")
            print(f"  Error percentage: {result['error_percentage']:.2f}%")
    
    # Calculate overall metrics
    mse = np.mean((y_test_actual - y_pred_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_actual - y_pred_actual))
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / (y_test_actual + 1e-8))) * 100
    
    print("\nOverall Model Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    return y_test_actual, y_pred_actual

def main(main_dir):
    print(f"Processing main directory: {main_dir}")
    train_features, test_features, test_file_names = process_main_directory(main_dir)
    
    print(f"Total training samples: {len(train_features)} (30 files from each program)")
    print(f"Total test samples: {len(test_features)} (2 files from each program)")
    
    if len(train_features) == 0 or len(test_features) == 0:
        print("Error: No valid training or test data found")
        return None
    
    # Prepare data for model
    X_train, y_train, X_test, y_test, y_scaler, input_size, is_log_transformed = prepare_data_for_model(train_features, test_features)
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=16)
    
    # Initialize enhanced model
    model = EnhancedLSTMModel(
        input_size=input_size,
        hidden_sizes=[128, 64, 32],  # Deeper architecture
        output_size=1,
        dropout_rate=0.3
    )
    
    # Define loss function and optimizer
    criterion = nn.HuberLoss(delta=1.0)  # More robust to outliers than MSE
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)  # Better optimizer with weight decay
    
    # Build and train model
    print("Building and training Enhanced LSTM model...")
    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        test_loader, 
        criterion, 
        optimizer, 
        num_epochs=150,  # More epochs 
        patience=20      # More patience for early stopping
    )
    
    # Evaluate model
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(model, X_test, y_test, y_scaler, test_file_names, is_log_transformed)
    
    return model, y_scaler, y_test_actual, y_pred_actual

if __name__ == "__main__":
    # Main directory containing subfolders for each program
    main_dir = "Output_Programs"
    
    # Run the main function to train and test
    model, y_scaler, y_test_actual, y_pred_actual = main(main_dir)
