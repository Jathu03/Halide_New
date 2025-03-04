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

def get_execution_time(file_path):
    """Extract the initial execution time or average from schedules_list."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Assuming data is nested under a function ID like "function1202328"
        for func_id, func_data in data.items():
            if "initial_execution_time" in func_data:
                return float(func_data["initial_execution_time"])
            
            if "schedules_list" in func_data:
                schedules = func_data["schedules_list"]
                if schedules and "execution_times" in schedules[0]:
                    # Use the average of the first schedule's execution times
                    exec_times = schedules[0]["execution_times"]
                    return float(np.mean(exec_times))
        
        print(f"Warning: No execution time found in {file_path}")
        return None
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {file_path}: {str(e)}")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred while processing {file_path}: {str(e)}")
        return None

def extract_features_from_file(file_path):
    """Extract features from the JSON file based on its structure."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    execution_time = get_execution_time(file_path)
    if execution_time is None:
        print(f"Warning: No execution time found in {file_path}")
        return None
    
    features = {'execution_time': execution_time}
    
    # Assuming data is nested under a function ID
    for func_id, func_data in data.items():
        if "program_annotation" in func_data:
            prog_annot = func_data["program_annotation"]
            features['memory_size'] = prog_annot.get("memory_size", 0)
            iterators = prog_annot.get("iterators", {})
            features['iterator_count'] = len(iterators)
            features['max_depth_iterators'] = max(
                (len(it.get("child_iterators", [])) for it in iterators.values()), default=0
            )
            computations = prog_annot.get("computations", {})
            features['computation_count'] = len(computations)
            features['reduction_count'] = sum(
                1 for comp in computations.values() if comp.get("comp_is_reduction", False)
            )
            features['access_count'] = sum(
                len(comp.get("accesses", [])) for comp in computations.values()
            )
        
        if "schedules_list" in func_data:
            schedules = func_data["schedules_list"]
            features['schedule_count'] = len(schedules)
            all_exec_times = []
            for sched in schedules:
                if "execution_times" in sched:
                    all_exec_times.extend(sched["execution_times"])
            if all_exec_times:
                features['avg_exec_time'] = float(np.mean(all_exec_times))
                features['min_exec_time'] = float(np.min(all_exec_times))
                features['max_exec_time'] = float(np.max(all_exec_times))
            features['tiling_count'] = sum(
                1 for sched in schedules for comp in sched.values() 
                if isinstance(comp, dict) and comp.get("tiling", {})
            )
            features['transformation_count'] = sum(
                sum(len(comp.get("transformations_list", [])) for comp in sched.values() if isinstance(comp, dict))
                for sched in schedules
            )
        
        if "exploration_trace" in func_data:
            trace = func_data["exploration_trace"]
            def count_nodes(trace_node):
                count = 1
                for child in trace_node.get("children", []):
                    count += count_nodes(child)
                return count
            features['exploration_node_count'] = count_nodes(trace)
    
    return features

def process_directory(directory_path):
    """Process all JSON files in the directory and split into train and test sets."""
    all_features = []
    file_names = []
    
    json_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.json')])
    
    if len(json_files) < 11:
        print(f"Error: Expected at least 11 files, found {len(json_files)} in {directory_path}")
        return None, None, None
    
    for filename in json_files:
        file_path = os.path.join(directory_path, filename)
        features = extract_features_from_file(file_path)
        if features is not None:
            all_features.append(features)
            file_names.append(filename)
    
    if len(all_features) < 11:
        print(f"Error: Only {len(all_features)} valid files found in {directory_path}")
        return None, None, None
    
    total_files = len(all_features)
    train_size = total_files - 10
    train_features = all_features[:train_size]
    test_features = all_features[train_size:]
    train_file_names = file_names[:train_size]
    test_file_names = file_names[train_size:]
    
    print(f"Processed {directory_path}: {len(train_features)} training files, {len(test_features)} test files")
    
    return train_features, test_features, test_file_names

def clean_and_transform_features(train_features, test_features):
    """Clean and transform features for improved model performance."""
    all_features_df = pd.DataFrame(train_features + test_features)
    all_features_df = all_features_df.fillna(0)
    
    constant_columns = [col for col in all_features_df.columns 
                        if col != 'execution_time' and all_features_df[col].nunique() == 1]
    all_features_df = all_features_df.drop(columns=constant_columns)
    print(f"Dropped {len(constant_columns)} constant columns")
    
    all_features_df['execution_time_log'] = np.log1p(all_features_df['execution_time'])
    
    numeric_cols = all_features_df.select_dtypes(include=['number']).columns
    all_features_df = all_features_df[numeric_cols]
    
    train_size = len(train_features)
    train_df = all_features_df.iloc[:train_size]
    test_df = all_features_df.iloc[train_size:]
    
    return train_df, test_df

def prepare_data_for_lstm(train_features, test_features):
    """Prepare training and testing data for the LSTM model."""
    train_df, test_df = clean_and_transform_features(train_features, test_features)
    
    y_train = train_df['execution_time_log'].values.reshape(-1, 1)
    y_test = test_df['execution_time_log'].values.reshape(-1, 1)
    X_train_df = train_df.drop(['execution_time', 'execution_time_log'], axis=1)
    X_test_df = test_df.drop(['execution_time', 'execution_time_log'], axis=1)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train_df)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(X_test_df)
    y_test_scaled = scaler_y.transform(y_test)
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    print(f"Input feature dimension: {X_train_scaled.shape[1]}")
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler_y, X_train_scaled.shape[1]

class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=1, dropout_rate=0.3):
        super(EnhancedLSTMModel, self).__init__()
        
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        self.lstm_layers.append(nn.LSTM(input_size, hidden_sizes[0], batch_first=True))
        self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        for i in range(1, len(hidden_sizes)):
            self.lstm_layers.append(nn.LSTM(hidden_sizes[i-1], hidden_sizes[i], batch_first=True))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        self.attention = nn.Linear(hidden_sizes[-1], 1)
        
        self.fc1 = nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2)
        self.bn1 = nn.BatchNorm1d(hidden_sizes[-1] // 2)
        self.fc2 = nn.Linear(hidden_sizes[-1] // 2, hidden_sizes[-1] // 4)
        self.bn2 = nn.BatchNorm1d(hidden_sizes[-1] // 4)
        self.output_layer = nn.Linear(hidden_sizes[-1] // 4, output_size)
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
        self.residual_adapter = nn.Linear(hidden_sizes[-1] // 2, hidden_sizes[-1] // 4)
        
    def attention_net(self, lstm_output):
        attn_weights = self.attention(lstm_output).squeeze(2)
        soft_attn_weights = torch.softmax(attn_weights, 1)
        context = torch.bmm(soft_attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context
        
    def forward(self, x):
        lstm_out = x
        for i, (lstm, dropout) in enumerate(zip(self.lstm_layers, self.dropout_layers)):
            lstm_out, _ = lstm(lstm_out)
            if i < len(self.lstm_layers) - 1:
                lstm_out = dropout(lstm_out)
        
        attn_output = self.attention_net(lstm_out)
        
        fc_out = self.fc1(attn_output)
        fc_out = self.bn1(fc_out)
        fc_out = self.leaky_relu(fc_out)
        
        residual = self.residual_adapter(fc_out)
        
        fc_out = self.fc2(fc_out)
        fc_out = self.bn2(fc_out)
        fc_out = self.leaky_relu(fc_out)
        
        fc_out = fc_out + residual
        
        output = self.output_layer(fc_out)
        
        return output

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=16):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=250, patience=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
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
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
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

def evaluate_model(model, X_test, y_test, y_scaler, file_names_test):
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
    y_test_actual = np.expm1(y_test_transformed)  # Reverse log1p transformation
    y_pred_actual = np.expm1(y_pred_transformed)
    
    print("\nEvaluation Results:")
    for i, file_name in enumerate(file_names_test):
        print(f"File: {file_name}")
        print(f"  Actual execution time: {y_test_actual[i][0]:.6f} seconds")
        print(f"  Predicted execution time: {y_pred_actual[i][0]:.6f} seconds")
        print(f"  Error percentage: {abs(y_test_actual[i][0] - y_pred_actual[i][0]) / y_test_actual[i][0] * 100:.2f}%")
    
    mse = np.mean((y_test_actual - y_pred_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_actual - y_pred_actual))
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / (y_test_actual + 1e-8))) * 100
    
    print("\nOverall Model Performance:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"MAPE: {mape:.2f}%")
    
    return y_test_actual, y_pred_actual

def main(main_dir):
    print(f"Processing directory: {main_dir}")
    train_features, test_features, test_file_names = process_directory(main_dir)
    
    if train_features is None or test_features is None:
        print("Error: Insufficient data to proceed")
        return None
    
    print(f"Total training samples: {len(train_features)}")
    print(f"Total test samples: {len(test_features)}")
    
    if len(train_features) == 0 or len(test_features) == 0:
        print("Error: No valid training or test data found")
        return None
    
    X_train, y_train, X_test, y_test, y_scaler, input_size = prepare_data_for_lstm(train_features, test_features)
    
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=16)
    
    model = EnhancedLSTMModel(
        input_size=input_size,
        hidden_sizes=[128, 64, 32],
        output_size=1,
        dropout_rate=0.3
    )
    
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    print("Building and training Enhanced LSTM model...")
    train_losses, val_losses = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=250, patience=20)
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(model, X_test, y_test, y_scaler, test_file_names)
    
    return model, y_scaler, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "Tiramisu"
    model, y_scaler, y_test_actual, y_pred_actual = main(main_dir)
    print("\nEnhanced model training and prediction completed!")
