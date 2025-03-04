import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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
        # Extract from program_annotation
        if "program_annotation" in func_data:
            prog_annot = func_data["program_annotation"]
            features['memory_size'] = prog_annot.get("memory_size", 0)
            
            # Iterator features
            iterators = prog_annot.get("iterators", {})
            features['iterator_count'] = len(iterators)
            features['max_depth_iterators'] = max(
                (len(it.get("child_iterators", [])) for it in iterators.values()), default=0
            )
            
            # Computation features
            computations = prog_annot.get("computations", {})
            features['computation_count'] = len(computations)
            features['reduction_count'] = sum(
                1 for comp in computations.values() if comp.get("comp_is_reduction", False)
            )
            features['access_count'] = sum(
                len(comp.get("accesses", [])) for comp in computations.values()
            )
        
        # Extract from schedules_list
        if "schedules_list" in func_data:
            schedules = func_data["schedules_list"]
            features['schedule_count'] = len(schedules)
            
            # Aggregate execution times stats
            all_exec_times = []
            for sched in schedules:
                if "execution_times" in sched:
                    all_exec_times.extend(sched["execution_times"])
            if all_exec_times:
                features['avg_exec_time'] = float(np.mean(all_exec_times))
                features['min_exec_time'] = float(np.min(all_exec_times))
                features['max_exec_time'] = float(np.max(all_exec_times))
            
            # Transformation and tiling features
            features['tiling_count'] = sum(
                1 for sched in schedules for comp in sched.values() 
                if isinstance(comp, dict) and comp.get("tiling", {})
            )
            features['transformation_count'] = sum(
                sum(len(comp.get("transformations_list", [])) for comp in sched.values() if isinstance(comp, dict))
                for sched in schedules
            )
        
        # Extract from exploration_trace
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
    
    # Get all JSON files in the directory
    json_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.json')])
    
    if len(json_files) < 5:  # Minimum threshold to ensure enough data for training
        print(f"Error: Expected at least 5 files, found {len(json_files)} in {directory_path}")
        return None, None, None
    
    # Process each file and extract features
    for filename in json_files:
        file_path = os.path.join(directory_path, filename)
        features = extract_features_from_file(file_path)
        if features is not None:
            all_features.append(features)
            file_names.append(filename)
    
    if len(all_features) < 5:
        print(f"Error: Only {len(all_features)} valid files found in {directory_path}")
        return None, None, None
    
    # Split into training (80%) and testing (20%)
    total_files = len(all_features)
    train_size = int(0.8 * total_files)
    train_features = all_features[:train_size]
    test_features = all_features[train_size:]
    train_file_names = file_names[:train_size]
    test_file_names = file_names[train_size:]
    
    print(f"Processed {directory_path}: {len(train_features)} training files, {len(test_features)} test files")
    
    return train_features, test_features, test_file_names

def prepare_data_for_lstm(train_features, test_features):
    """Prepare training and testing data for the LSTM model."""
    all_features_df = pd.DataFrame(train_features + test_features)
    if len(all_features_df) <= 5:
        raise ValueError("Not enough data samples to train the model")
    
    X = all_features_df.drop('execution_time', axis=1)
    y = all_features_df['execution_time']
    X = X.fillna(0)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    train_size = len(train_features)
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size].values.reshape(-1, 1)
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:].values.reshape(-1, 1)
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler_y, X_train_scaled.shape[1]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=64, hidden_size2=32, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_size)
        
    def forward(self, x):
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout1(lstm_out1)
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2[:, -1, :])
        fc1_out = self.relu(self.fc1(lstm_out2))
        output = self.fc2(fc1_out)
        return output

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=100, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
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
    
    y_test_actual = y_scaler.inverse_transform(y_test)
    y_pred_actual = y_scaler.inverse_transform(y_pred_scaled)
    
    print("\nEvaluation Results:")
    for i, file_name in enumerate(file_names_test):
        print(f"File: {file_name}")
        print(f"  Actual execution time: {y_test_actual[i][0]:.6f} seconds")
        print(f"  Predicted execution time: {y_pred_actual[i][0]:.6f} seconds")
    
    mse = np.mean((y_test_actual - y_pred_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_actual - y_pred_actual))
    
    print("\nOverall Model Performance:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    
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
    
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32)
    model = LSTMModel(input_size=input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Building and training LSTM model...")
    train_losses, val_losses = train_model(model, train_loader, test_loader, criterion, optimizer)
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(model, X_test, y_test, y_scaler, test_file_names)
    
    return model, y_scaler, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "Tiramisu"  # Directory containing all JSON files with no subfolders
    model, y_scaler, y_test_actual, y_pred_actual = main(main_dir)
    print("\nModel training and prediction completed!")
