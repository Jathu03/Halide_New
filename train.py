import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def get_execution_time(file_path):
    try:
        # Open the file in binary mode to handle raw bytes
        with open(file_path, 'rb') as f:
            # Read the raw bytes and decode, replacing null characters
            raw_content = f.read()
            # Decode bytes to string, replacing null characters with an empty string
            content = raw_content.decode('utf-8', errors='replace').replace('\0', '')
            # Parse the cleaned content as JSON
            data = json.loads(content)
        
        # Check if 'programming_details' exists
        if 'programming_details' not in data:
            print(f"Error: 'programming_details' key not found in {file_path}")
            return None
        
        # Access the 'Schedules' list within 'programming_details'
        
        schedules = data["scheduling_data"]
        # Look for the 'total_execution_time_ms' entry
        for item in schedules:
            if isinstance(item, dict) and item.get('name') == 'total_execution_time_ms':
                execution_time = item.get('value')
                if execution_time is not None:
                    return float(execution_time)  # Ensure it's returned as a float
        
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

# Function to extract features from a single JSON file
def extract_features_from_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract execution time (target variable)
    execution_time = get_execution_time(file_path)
    
    if execution_time is None:
        print(f"Warning: No execution time found in {file_path}")
        return None
    
    # Extract programming details (node and edge features)
    nodes_features = []
    edges_features = []
    
    # Look for programming_details
    programming_details = None
    for key, value in data.items():
        if key == "programming_details":
            programming_details = value
            break
    
    if programming_details:
        # Process nodes
        if 'Nodes' in programming_details:
            for node in programming_details['Nodes']:
                node_feature = {}
                node_feature['Name'] = node.get('Name', '')
                
                # Extract op histogram if available
                if 'Details' in node and 'Op histogram' in node['Details']:
                    op_hist = node['Details']['Op histogram']
                    for op_line in op_hist:
                        parts = op_line.strip().split(':')
                        if len(parts) == 2:
                            op_name = parts[0].strip()
                            op_count = int(parts[1].strip())
                            node_feature[f'op_{op_name.lower()}'] = op_count
                
                nodes_features.append(node_feature)
        
        # Process edges
        if 'Edges' in programming_details:
            for edge in programming_details['Edges']:
                edge_feature = {}
                edge_feature['From'] = edge.get('From', '')
                edge_feature['To'] = edge.get('To', '')
                edge_feature['Name'] = edge.get('Name', '')
                
                edges_features.append(edge_feature)
    
    # Extract scheduling details
    scheduling_features = []
    
    # Look for scheduling_data or use programming_details for Schedules
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
    
    # Create a feature dictionary
    features = {
        'execution_time': execution_time,
        'nodes_count': len(nodes_features),
        'edges_count': len(edges_features),
        'scheduling_count': len(scheduling_features)
    }
    
    # Add aggregated node features
    op_counts = {}
    for node in nodes_features:
        for key, value in node.items():
            if key.startswith('op_'):
                op_counts[key] = op_counts.get(key, 0) + value
    
    features.update(op_counts)
    
    # Add important scheduling features (aggregate or take first few entries)
    if scheduling_features:
        # Common scheduling metrics to extract
        important_metrics = [
            'bytes_at_production', 'bytes_at_realization', 'bytes_at_root', 'bytes_at_task',
            'inner_parallelism', 'outer_parallelism', 'num_productions', 'num_realizations',
            'num_scalars', 'num_vectors', 'points_computed_total', 'working_set'
        ]
        
        # Take the first scheduling feature for these metrics
        if scheduling_features and scheduling_features[0]:
            for metric in important_metrics:
                if metric in scheduling_features[0]:
                    features[f'sched_{metric}'] = scheduling_features[0][metric]
        
        # Aggregate some metrics across all scheduling features
        total_bytes_at_production = sum(sf.get('bytes_at_production', 0) for sf in scheduling_features if isinstance(sf, dict))
        total_vectors = sum(sf.get('num_vectors', 0) for sf in scheduling_features if isinstance(sf, dict))
        total_parallelism = sum(sf.get('inner_parallelism', 0) * sf.get('outer_parallelism', 1) for sf in scheduling_features if isinstance(sf, dict))
        
        features['total_bytes_at_production'] = total_bytes_at_production
        features['total_vectors'] = total_vectors
        features['total_parallelism'] = total_parallelism
    
    return features

# Function to process all JSON files in a directory
def process_all_files(directory_path):
    all_features = []
    file_names = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            features = extract_features_from_file(file_path)
            
            if features is not None:
                all_features.append(features)
                file_names.append(filename)
    
    return all_features, file_names

# Function to prepare data for LSTM
def prepare_data_for_lstm(all_features, test_indices=None):
    # Convert to dataframe
    df = pd.DataFrame(all_features)
    
    # Check if we have enough data
    if len(df) <= 5:  # arbitrary small number
        raise ValueError("Not enough data samples to train the model")
    
    # Separate features and target
    X = df.drop('execution_time', axis=1)
    y = df['execution_time']
    
    # Handle missing values (replace with 0)
    X = X.fillna(0)
    
    # Normalize features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # If test_indices is provided, split data accordingly
    if test_indices is not None:
        test_mask = np.zeros(len(df), dtype=bool)
        test_mask[test_indices] = True
        train_mask = ~test_mask
        
        X_train = X[train_mask]
        y_train = y[train_mask].values.reshape(-1, 1)
        X_test = X[test_mask]
        y_test = y[test_mask].values.reshape(-1, 1)
        
        # Fit scalers on training data only
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)
        
        # Transform test data using the same scalers
        X_test_scaled = scaler_X.transform(X_test)
        y_test_scaled = scaler_y.transform(y_test)
    else:
        # Random split if test_indices not provided
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Reshape y for scaling
        y_train = y_train.values.reshape(-1, 1)
        y_test = y_test.values.reshape(-1, 1)
        
        # Scale the data
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_test_scaled = scaler_y.transform(y_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    # Reshape for LSTM [batch, seq_len, features]
    X_train_tensor = X_train_tensor.unsqueeze(1)  # Add sequence dimension of 1
    X_test_tensor = X_test_tensor.unsqueeze(1)   # Add sequence dimension of 1
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler_y, X_train_scaled.shape[1]

# Define the LSTM model class
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
        # Input shape: [batch_size, seq_len, input_size]
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout1(lstm_out1)
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2[:, -1, :])  # Take the last time step output
        fc1_out = self.relu(self.fc1(lstm_out2))
        output = self.fc2(fc1_out)
        return output

# Function to create data loaders
def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=4):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Function to train the model
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=100, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model.to(device)
    
    # To track the best validation loss
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
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
        
        # Early stopping
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
    
    # If we completed all epochs, use the best model
    if best_model_state is not None and epochs_no_improve > 0:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses

# Function to evaluate the model
def evaluate_model(model, X_test, y_test, y_scaler, file_names_test=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Move data to device
    X_test = X_test.to(device)
    
    # Predict
    with torch.no_grad():
        y_pred_scaled = model(X_test)
    
    # Convert to numpy
    y_pred_scaled = y_pred_scaled.cpu().numpy()
    y_test = y_test.cpu().numpy()
    
    # Inverse transform to get actual values
    y_test_actual = y_scaler.inverse_transform(y_test)
    y_pred_actual = y_scaler.inverse_transform(y_pred_scaled)
    
    # Calculate metrics
    mse = np.mean((y_test_actual - y_pred_actual) ** 2)
    mae = np.mean(np.abs(y_test_actual - y_pred_actual))
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    
    return mse, mae, mape, y_test_actual, y_pred_actual

# Main function
def main(data_dir, test_file_indices=None):
    # Process all files
    print(f"Processing files in directory: {data_dir}")
    all_features, file_names = process_all_files(data_dir)
    print(f"Processed {len(all_features)} files")
    
    # If test_file_indices is provided, use it; otherwise, use random split
    if test_file_indices is not None:
        X_train, y_train, X_test, y_test, y_scaler, input_size = prepare_data_for_lstm(all_features, test_file_indices)
        file_names_test = [file_names[i] for i in test_file_indices]
    else:
        X_train, y_train, X_test, y_test, y_scaler, input_size = prepare_data_for_lstm(all_features)
        file_names_test = None
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=4)
    
    # Initialize the model
    model = LSTMModel(input_size=input_size)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("Building and training LSTM model...")
    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        test_loader, 
        criterion, 
        optimizer, 
        num_epochs=100,
        patience=10
    )
    
    # Evaluate the model
    print("\nEvaluating model:")
    mse, mae, mape, y_test_actual, y_pred_actual = evaluate_model(model, X_test, y_test, y_scaler, file_names_test)
    
    return model, y_scaler, mse, mae, mape

# Example usage
if __name__ == "__main__":
    # Directory containing the JSON files
    data_dir = "Output_Programs/program_50001"
    
    # Specify the indices of test files (e.g., the last two files)
    # For 32 files (0-31), the last two would be indices 30 and 31
    test_file_indices = [30, 31]
    
    # Run the main function
    model, y_scaler, mse, mae, mape = main(data_dir, test_file_indices)
    
    print("\nModel training completed!")
    print(f"Final Mean Squared Error: {mse:.2f}")
    print(f"Final Mean Absolute Error: {mae:.2f}")
    print(f"Final Mean Absolute Percentage Error: {mape:.2f}%")
