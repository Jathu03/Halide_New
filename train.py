import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Step 1: Load JSON Files from Directory
def load_json_files(directory):
    data = []
    program_names = []
    
    for program_folder in os.listdir(directory):
        program_path = os.path.join(directory, program_folder)
        if os.path.isdir(program_path):
            program_data = []
            for json_file in os.listdir(program_path):
                if json_file.endswith('.json'):
                    file_path = os.path.join(program_path, json_file)
                    try:
                        with open(file_path, 'r') as f:
                            json_data = json.load(f)
                            # No need to check if it's a list; just append the raw JSON object
                            program_data.append(json_data)
                    except json.JSONDecodeError as e:
                        print(f"Skipping {file_path}: Invalid JSON ({e})")
                    except Exception as e:
                        print(f"Skipping {file_path}: Error ({e})")
            if program_data:  # Only append if some data was successfully loaded
                data.append(program_data)
                program_names.append(program_folder)
            else:
                print(f"Warning: No valid JSON files loaded from {program_folder}")
    
    if not data:
        raise ValueError("No valid data loaded from Output_Programs folder")
    return data, program_names

# Step 2: Feature Extraction
def extract_features(json_data):
    features = []
    
    # Extract from Edges (e.g., number of edges, footprint size, Jacobian complexity)
    edges = json_data.get('programming_details', {}).get('Edges', [])
    num_edges = len(edges)
    footprint_sizes = [len(edge['Details']['Footprint']) for edge in edges if 'Details' in edge and 'Footprint' in edge['Details']]
    jacobian_sizes = [len(edge['Details']['Load Jacobians']) for edge in edges if 'Details' in edge and 'Load Jacobians' in edge['Details']]
    
    features.extend([
        num_edges,
        np.mean(footprint_sizes) if footprint_sizes else 0,
        np.mean(jacobian_sizes) if jacobian_sizes else 0
    ])
    
    # Extract from Nodes (e.g., memory patterns, op histogram)
    nodes = json_data.get('programming_details', {}).get('Nodes', [])
    num_nodes = len(nodes)
    memory_patterns = [sum(map(int, node['Details']['Memory access patterns'][0].split()[1:])) 
                       for node in nodes if node.get('Details', {}).get('Memory access patterns')]
    op_histogram = [sum(int(op.split()[-1]) for op in node['Details']['Op histogram']) 
                    for node in nodes if node.get('Details', {}).get('Op histogram')]
    
    features.extend([
        num_nodes,
        np.mean(memory_patterns) if memory_patterns else 0,
        np.mean(op_histogram) if op_histogram else 0
    ])
    
    # Placeholder for scheduling features (adjust based on actual structure)
    sched_features = []
    for item in json_data.get('schedule_feature', []):
        if isinstance(item, dict) and 'Details' in item and 'scheduling_feature' in item['Details']:
            sched_features.append(item['Details']['scheduling_feature'])
    if sched_features:
        avg_sched_features = {k: np.mean([sf[k] for sf in sched_features]) 
                              for k in sched_features[0].keys()}
        features.extend(list(avg_sched_features.values()))
    
    # Target: Execution Time
    execution_time = None
    for item in json_data.get('schedule_feature', []):
        if isinstance(item, dict) and item.get('name') == 'total_execution_time_ms':
            execution_time = item.get('value', 0)
            break
    if execution_time is None:
        execution_time = 0  # Default if not found
    
    return np.array(features, dtype=float), execution_time

# Step 3: Prepare Data for LSTM
def prepare_lstm_data(data):
    X, y = [], []
    feature_dim = None
    
    for program_data in data:
        program_X, program_y = [], []
        for schedule in program_data:
            features, exec_time = extract_features(schedule)
            if feature_dim is None:
                feature_dim = len(features)
            elif len(features) != feature_dim:
                # Pad or truncate to ensure consistent feature dimension
                features = np.pad(features, (0, feature_dim - len(features)), 'constant')[:feature_dim]
            program_X.append(features)
            program_y.append(exec_time)
        
        if program_X and program_y:  # Ensure non-empty sequences
            X.append(program_X)
            y.append(program_y)
    
    if not X or not y:
        raise ValueError("No valid sequences prepared from the data")
    
    # Convert to numpy arrays
    X = np.array(X)  # Shape: (num_programs, num_schedules, feature_dim)
    y = np.array(y)  # Shape: (num_programs, num_schedules)
    
    # Normalize features
    scaler_X = MinMaxScaler()
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_normalized = scaler_X.fit_transform(X_reshaped).reshape(X.shape)
    
    # Normalize target
    scaler_y = MinMaxScaler()
    y_reshaped = y.reshape(-1, 1)
    y_normalized = scaler_y.fit_transform(y_reshaped).reshape(y.shape)
    
    return X_normalized, y_normalized, scaler_X, scaler_y

# Step 4: Build and Train LSTM Model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)  # Predict execution time
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Main Execution
def predict_halide_speedup(data_dir):
    # Load data
    data, program_names = load_json_files(data_dir)
    print(f"Loaded data from {len(program_names)} programs")
    
    # Prepare data
    X, y, scaler_X, scaler_y = prepare_lstm_data(data)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build model
    input_shape = (X.shape[1], X.shape[2])  # (num_schedules, feature_dim)
    model = build_lstm_model(input_shape)
    
    # Train model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
    
    # Evaluate model
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {test_loss}")
    
    # Predict and inverse transform
    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_test.shape)
    y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
    
    # Print example predictions
    for i in range(min(5, len(y_test))):
        print(f"Program {i+1}: True Time: {y_test_rescaled[i,0]:.2f} ms, Predicted Time: {y_pred_rescaled[i,0]:.2f} ms")
    
    # Save model
    model.save('lstm_execution_time_predictor.h5')

if __name__ == "__main__":
    predict_halide_speedup(data_dir="Output_Programs")
