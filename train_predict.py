import torch
import numpy as np

# Load model and scalers
model = AdvancedExecutionTimePredictor()
model.load_state_dict(torch.load('execution_predictor_advanced.pt'))
model.eval()
X_scaler_mean = np.load('X_scaler_mean.npy')
X_scaler_scale = np.load('X_scaler_scale.npy')
y_scaler_mean = np.load('y_scaler_mean.npy')
y_scaler_scale = np.load('y_scaler_scale.npy')

# Example: New sequence
new_sequence = np.load('halide_data_0.npy')[np.newaxis, :]
new_sequence_scaled = (new_sequence - X_scaler_mean) / X_scaler_scale
new_sequence_tensor = torch.tensor(new_sequence_scaled, dtype=torch.float32)

# Predict
with torch.no_grad():
    pred_scaled = model(new_sequence_tensor).numpy()
    pred_seconds = (pred_scaled * y_scaler_scale) + y_scaler_mean
    print("Predicted Execution Time (seconds):", pred_seconds[0][0])
