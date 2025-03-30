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
import random

# [All existing functions remain unchanged until the main function]

def create_schedule_representation(model, schedule_features, feature_names, scaler_X_path='scaler_X.json', scaler_y_path='scaler_y.json'):
    """
    Create a schedule representation using the trained model.
    
    Args:
        model: Trained PyTorch model
        schedule_features: Dictionary containing the schedule features
        feature_names: List of feature names expected by the model
        scaler_X_path: Path to feature scaler parameters
        scaler_y_path: Path to target scaler parameters
    
    Returns:
        Dictionary containing the schedule representation and original features
    """
    # Load feature scaler parameters
    with open(scaler_X_path, 'r') as f:
        scaler_X_data = json.load(f)
    
    # Create DataFrame from input features
    input_df = pd.DataFrame([schedule_features])
    
    # Ensure all expected features are present and fill missing with 0
    for feature in scaler_X_data['feature_names']:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    # Reorder columns to match training data
    input_df = input_df[scaler_X_data['feature_names']]
    
    # Apply same scaling as training data
    means = np.array(scaler_X_data['means'])
    scales = np.array(scaler_X_data['scales'])
    X_scaled = (input_df.values - means) / scales
    
    # Convert to tensor and add batch dimension
    X_tensor = torch.FloatTensor(X_scaled).unsqueeze(1)
    
    # Get model representation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        # Get the output of the last LSTM layer before attention
        lstm_out = X_tensor.to(device)
        
        # Process through each LSTM layer
        for i, (lstm, dropout) in enumerate(zip(model.lstm_layers, model.dropout_layers)):
            batch_size = lstm_out.size(0)
            hidden_size = model.hidden_sizes[i]
            h_0 = torch.zeros(1, batch_size, hidden_size, device=device)
            c_0 = torch.zeros(1, batch_size, hidden_size, device=device)
            
            lstm_out, _ = lstm(lstm_out, (h_0, c_0))
            if i < len(model.lstm_layers) - 1:
                lstm_out = dropout(lstm_out)
        
        # Get the attention weights
        attn_weights = model.attention(lstm_out).squeeze(2)
        soft_attn_weights = torch.softmax(attn_weights, 1)
        
        # Get the context vector
        context = torch.bmm(soft_attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        
        # Get the final representation before output layer
        fc_out = model.fc_layers[0](context)
        fc_out = model.bn_layers[0](fc_out)
        fc_out = model.leaky_relu(fc_out)
        
        # Get the compressed representation (128-dim vector)
        representation = fc_out.cpu().numpy().flatten()
    
    # Create output dictionary
    output = {
        'original_features': schedule_features,
        'representation': representation.tolist(),
        'attention_weights': soft_attn_weights.cpu().numpy().flatten().tolist()
    }
    
    return output

def main(main_dir):
    print(f"Processing main directory: {main_dir}")
    train_features, test_features, test_file_names = process_main_directory(main_dir)
    
    print(f"Total training samples: {len(train_features)} (randomly selected)")
    print(f"Total test samples: {len(test_features)} (50 randomly selected)")
    
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
        hidden_sizes=[128, 64, 32],
        output_size=1,
        dropout_rate=0.3
    )
    
    # Define loss function and optimizer
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Build and train model
    print("Building and training Enhanced LSTM model...")
    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        test_loader, 
        criterion, 
        optimizer, 
        num_epochs=150,
        patience=20
    )
    
    # Evaluate model
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(model, X_test, y_test, y_scaler, test_file_names, is_log_transformed)
    
    # Save the trained model as a .pt file using TorchScript
    print("\nSaving the trained model as 'lstm_model.pt'...")
    model.eval()  # Set the model to evaluation mode
    
    # Determine the device the model is on
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")
    
    try:
        # Create sample input and move it to the same device as the model
        sample_input = torch.randn(1, 1, input_size).to(device)  # [batch_size, sequence_length, input_size]
        
        # Trace the model with the sample input
        traced_model = torch.jit.trace(model, sample_input)
        
        # Save the traced model to a .pt file
        traced_model.save("lstm_model.pt")
        print("Model successfully saved as 'lstm_model.pt'")
    except Exception as e:
        print(f"Error saving the model: {str(e)}")
    
    # Get feature names from the scaler file
    with open('scaler_X.json', 'r') as f:
        scaler_data = json.load(f)
    feature_names = scaler_data['feature_names']
    
    # Example: Create representation for one schedule
    if len(test_features) > 0:
        sample_schedule = test_features[0]
        representation = create_schedule_representation(model, sample_schedule, feature_names)
        with open('schedule_representation.json', 'w') as f:
            json.dump(representation, f, indent=2)
        print("\nSaved schedule representation to 'schedule_representation.json'")
    
    return model, y_scaler, y_test_actual, y_pred_actual

if __name__ == "__main__":
    # Main directory containing subfolders for each program
    main_dir = "synthetic_data"
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Run the main function to train and test
    model, y_scaler, y_test_actual, y_pred_actual = main(main_dir)
