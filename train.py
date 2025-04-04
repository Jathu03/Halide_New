import torch
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Define the model class (copy from your original script)
class EnhancedLSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=1, dropout_rate=0.3):
        super(EnhancedLSTMModel, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.lstm_layers = torch.nn.ModuleList()
        self.dropout_layers = torch.nn.ModuleList()
        self.lstm_layers.append(torch.nn.LSTM(input_size, hidden_sizes[0], batch_first=True))
        self.dropout_layers.append(torch.nn.Dropout(dropout_rate))
        for i in range(1, len(hidden_sizes)):
            self.lstm_layers.append(torch.nn.LSTM(hidden_sizes[i-1], hidden_sizes[i], batch_first=True))
            self.dropout_layers.append(torch.nn.Dropout(dropout_rate))
        self.attention = torch.nn.Linear(hidden_sizes[-1], 1)
        self.fc_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()
        self.fc_layers.append(torch.nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2))
        self.bn_layers.append(torch.nn.BatchNorm1d(hidden_sizes[-1] // 2))
        self.fc_layers.append(torch.nn.Linear(hidden_sizes[-1] // 2, hidden_sizes[-1] // 4))
        self.bn_layers.append(torch.nn.BatchNorm1d(hidden_sizes[-1] // 4))
        self.output_layer = torch.nn.Linear(hidden_sizes[-1] // 4, output_size)
        self.relu = torch.nn.ReLU()
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.has_residual = (hidden_sizes[-1] // 4 == hidden_sizes[-1] // 2)
        if not self.has_residual:
            self.residual_adapter = torch.nn.Linear(hidden_sizes[-1] // 2, hidden_sizes[-1] // 4)

    def attention_net(self, lstm_output):
        attn_weights = self.attention(lstm_output).squeeze(2)
        soft_attn_weights = torch.softmax(attn_weights, 1)
        context = torch.bmm(soft_attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context

    def forward(self, x):
        lstm_out = x
        for i, (lstm, dropout) in enumerate(zip(self.lstm_layers, self.dropout_layers)):
            hidden_size = self.hidden_sizes[i]
            h_0 = torch.zeros(1, x.size(0), hidden_size)
            c_0 = torch.zeros(1, x.size(0), hidden_size)
            lstm_out, _ = lstm(lstm_out, (h_0, c_0))
            if i < len(self.lstm_layers) - 1:
                lstm_out = dropout(lstm_out)
        attn_output = self.attention_net(lstm_out)
        fc_out = self.fc_layers[0](attn_output)
        fc_out = self.bn_layers[0](fc_out)
        fc_out = self.leaky_relu(fc_out)
        residual = fc_out
        if not self.has_residual:
            residual = self.residual_adapter(residual)
        fc_out = self.fc_layers[1](fc_out)
        fc_out = self.bn_layers[1](fc_out)
        fc_out = self.leaky_relu(fc_out)
        fc_out = fc_out + residual
        output = self.output_layer(fc_out)
        return output

# Load the model
device = torch.device('cpu')
model = torch.jit.load('lstm_model.pt', map_location=device)
model.eval()

# Load scalers
y_scaler = joblib.load('y_scaler.pkl')
with open('scaler_X.json', 'r') as f:
    scaler_X_data = json.load(f)
with open('scaler_y.json', 'r') as f:
    scaler_y_data = json.load(f)

# Example inference with dummy data (replace with your features)
input_size = len(scaler_X_data['feature_names'])  # From training
dummy_input = torch.randn(1, 1, input_size).to(device)
with torch.no_grad():
    pred_scaled = model(dummy_input).numpy()
pred_transformed = y_scaler.inverse_transform(pred_scaled)
if scaler_y_data['is_log_transformed']:
    pred_actual = np.expm1(pred_transformed)
else:
    pred_actual = pred_transformed
print(f"Predicted execution time: {pred_actual[0][0]:.2f} ms")
