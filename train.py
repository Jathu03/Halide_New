import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
import json

# 1. Feature Extraction Function
def extract_features(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    features = []
    
    # Extract Edge Features (numerical values from Load Jacobians)
    edge_features = []
    for edge in data['programming_details']['Edges']:
        jacobian = edge['Details']['Load Jacobians']
        # Convert Jacobian strings to numerical values
        jacobian_vals = []
        for row in jacobian:
            nums = [float(x) for x in row.split() if x not in ['_', '0', '1']]
            jacobian_vals.extend(nums)
        edge_features.extend(jacobian_vals)
    
    # Extract Node Features (numerical values from Op histogram)
    node_features = []
    for node in data['programming_details']['Nodes']:
        histogram = node['Details']['Op histogram']
        hist_vals = []
        for line in histogram:
            if ':' in line:
                val = float(line.split(':')[1].strip())
                hist_vals.append(val)
        node_features.extend(hist_vals)
    
    # Extract Scheduling Features
    scheduling_features = []
    for schedule in data['programming_details']['Schedules']:
        if isinstance(schedule, dict) and 'Details' in schedule:
            sched_data = schedule['Details']['scheduling_feature']
            sched_vals = [float(v) for v in sched_data.values()]
            scheduling_features.extend(sched_vals)
    
    # Combine all features
    features.extend(edge_features)
    features.extend(node_features)
    features.extend(scheduling_features)
    
    # Get execution time (y_data)
    execution_time = next(item['value'] for item in data['programming_details']['Schedules'] 
                         if item.get('name') == 'total_execution_time_ms')
    
    return np.array(features), execution_time

# 2. Load and Process All Files
def load_data(folder_path):
    X_data = []
    y_data = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            features, exec_time = extract_features(file_path)
            X_data.append(features)
            y_data.append(exec_time)
    
    # Pad sequences to same length
    max_length = max(len(x) for x in X_data)
    X_data_padded = np.array([np.pad(x, (0, max_length - len(x)), 'constant') 
                            for x in X_data])
    
    return X_data_padded, np.array(y_data)

# 3. Prepare Sequences for LSTM
def create_sequences(X, y, sequence_length=5):
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:(i + sequence_length)])
        y_seq.append(y[i + sequence_length])
    return np.array(X_seq), np.array(y_seq)

# 4. Main Training Function
def train_lstm_model(folder_path):
    # Load and preprocess data
    X_data, y_data = load_data(folder_path)
    
    # Normalize features
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_data)
    
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y_data.reshape(-1, 1))
    
    # Create sequences
    sequence_length = 5
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, sequence_length)
    
    # Split into train and test sets
    train_size = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:train_size], X_seq[train_size:]
    y_train, y_test = y_seq[:train_size], y_seq[train_size:]
    
    # Build LSTM model
    model = Sequential([
        LSTM(100, input_shape=(sequence_length, X_scaled.shape[1]), return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate model
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss}")
    
    # Save model and scalers
    model.save('lstm_schedule_predictor.h5')
    np.save('scaler_X.npy', scaler_X)
    np.save('scaler_y.npy', scaler_y)
    
    return model, scaler_X, scaler_y

# 5. Prediction Function
def predict_execution_time(model, scaler_X, scaler_y, new_file_path, sequence_length=5):
    # Load and preprocess new data
    features, _ = extract_features(new_file_path)
    max_length = scaler_X.data_max_.shape[0]
    features_padded = np.pad(features, (0, max_length - len(features)), 'constant')
    features_scaled = scaler_X.transform(features_padded.reshape(1, -1))
    
    # Create sequence (assuming we have previous data points)
    # For single prediction, we'll need to provide a sequence
    # Here we'll repeat the single input for demonstration
    sequence = np.repeat(features_scaled, sequence_length, axis=0)[np.newaxis, :]
    
    # Make prediction
    pred_scaled = model.predict(sequence)
    prediction = scaler_y.inverse_transform(pred_scaled)
    
    return prediction[0][0]

# Main execution
if __name__ == "__main__":
    folder_path = 'Output_Programs/program_50001'  # Replace with actual path
    model, scaler_X, scaler_y = train_lstm_model(folder_path)
    
    # Example prediction
    new_file = 'path/to/new/schedule.json'
    predicted_time = predict_execution_time(model, scaler_X, scaler_y, new_file)
    print(f"Predicted Execution Time: {predicted_time} ms")
