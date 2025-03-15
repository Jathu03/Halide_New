import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import random
import matplotlib.pyplot as plt

def get_execution_time(schedule_data):
    if "execution_times" in schedule_data:
        exec_times = schedule_data["execution_times"]
        return float(np.mean(exec_times))
    print("Warning: No execution times found in schedule")
    return None

def extract_features_from_file(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None
    
    all_features = []
    
    for func_id, func_data in data.items():
        if "program_annotation" not in func_data or "schedules_list" not in func_data:
            print(f"Warning: Missing required fields in {file_path} for {func_id}")
            continue
        
        prog_annot = func_data["program_annotation"]
        iterators = prog_annot.get("iterators", {})
        computations = prog_annot.get("computations", {})
        
        loop_ranges = []
        for it in iterators.values():
            lower = it.get("lower_bound")
            upper = it.get("upper_bound")
            if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                loop_ranges.append(upper - lower)
        
        # Base features (non-sequential, kept for context)
        base_features = {
            'memory_size': prog_annot.get("memory_size", 0),
            'iterator_count': len(iterators),
            'max_depth_iterators': max((len(it.get("child_iterators", [])) for it in iterators.values()), default=0),
            'computation_count': len(computations),
            'reduction_count': sum(1 for comp in computations.values() if comp.get("comp_is_reduction", False)),
            'access_count': sum(len(comp.get("accesses", [])) for comp in computations.values()),
            'avg_loop_range': float(np.mean(loop_ranges)) if loop_ranges else 0,
        }
        
        schedules = func_data["schedules_list"]
        for idx, schedule in enumerate(schedules):
            execution_time = get_execution_time(schedule)
            if execution_time is None or execution_time <= 0:
                continue
            
            features = base_features.copy()
            features['execution_time'] = execution_time
            features['log_execution_time'] = np.log1p(execution_time)
            
            # Sequential transformation features
            transformation_seq = []
            for comp_key, comp_data in schedule.items():
                if isinstance(comp_data, dict) and "transformations_list" in comp_data:
                    for transform in comp_data["transformations_list"]:
                        transform_features = [0, 0, 0, 0]  # [tile, unroll, parallel, factor]
                        if isinstance(transform, str):  # Assuming string format
                            if "tile" in transform.lower():
                                transform_features[0] = 1
                                factors = [int(x) for x in transform.split('[')[1].split(']')[0].split(',')]
                                transform_features[3] = factors[0]  # First tiling factor
                            elif "unroll" in transform.lower():
                                transform_features[1] = 1
                                transform_features[3] = int(transform.split('[')[1].split(']')[0])
                            elif "parallel" in transform.lower():
                                transform_features[2] = 1
                        transformation_seq.append(transform_features)
            
            features['transformations'] = transformation_seq
            all_features.append(features)
    
    return all_features if all_features else None

def process_directory(directory_path):
    all_features = []
    file_names = []
    
    json_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.json')])
    
    for filename in json_files:
        file_path = os.path.join(directory_path, filename)
        features_list = extract_features_from_file(file_path)
        if features_list is not None:
            all_features.extend(features_list)
            file_names.extend([f"{filename}_schedule_{i}" for i in range(len(features_list))])
    
    if len(all_features) < 60:
        print(f"Error: Only {len(all_features)} valid schedules found in {directory_path}")
        return None, None, None
    
    combined = list(zip(all_features, file_names))
    random.shuffle(combined)
    all_features, file_names = zip(*combined)
    
    train_features = list(all_features[:-10])
    test_features = list(all_features[-10:])
    train_file_names = list(file_names[:-10])
    test_file_names = list(file_names[-10:])
    
    print(f"Processed {directory_path}: {len(train_features)} training schedules, {len(test_features)} test schedules")
    
    return train_features, test_features, test_file_names

def prepare_data_for_model(train_features, test_features, max_seq_len=10):
    all_features = train_features + test_features
    X_seq = []
    X_base = []
    y = []
    
    # Define base feature keys to keep
    base_keys = ['memory_size', 'iterator_count', 'max_depth_iterators', 
                 'computation_count', 'reduction_count', 'access_count', 'avg_loop_range']
    
    for feat in all_features:
        # Sequential data (transformations)
        seq = feat.pop("transformations")
        padded_seq = np.zeros((max_seq_len, 4))  # [tile, unroll, parallel, factor]
        for i, t in enumerate(seq[:max_seq_len]):
            padded_seq[i] = t
        X_seq.append(padded_seq)
        
        # Base features (non-sequential)
        base_feat = [feat[k] for k in base_keys]
        X_base.append(base_feat)
        
        # Target
        y.append(feat["log_execution_time"])
    
    X_seq = np.array(X_seq)  # [n_samples, max_seq_len, 4]
    X_base = np.array(X_base)  # [n_samples, n_base_features]
    y = np.array(y).reshape(-1, 1)
    
    # Scale base features
    scaler_X_base = StandardScaler()
    X_base_scaled = scaler_X_base.fit_transform(X_base)
    
    # Combine sequential and base features
    # Repeat base features across sequence length for concatenation
    X_base_scaled_expanded = np.repeat(X_base_scaled[:, np.newaxis, :], max_seq_len, axis=1)
    X_combined = np.concatenate([X_seq, X_base_scaled_expanded], axis=2)  # [n_samples, max_seq_len, 4 + n_base_features]
    
    # Split back into train/test
    train_size = len(train_features)
    X_train = X_combined[:train_size]
    X_test = X_combined[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    # Scale target
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    print(f"Input shape: {X_train.shape}")
    return (torch.FloatTensor(X_train), torch.FloatTensor(y_train_scaled),
            torch.FloatTensor(X_test), torch.FloatTensor(y_test_scaled),
            scaler_y, X_train.shape[2])

class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], output_size=1, dropout_rate=0.5):
        super(EnhancedLSTMModel, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True, bidirectional=True, dropout=0.2)
        self.lstm2 = nn.LSTM(hidden_sizes[0]*2, hidden_sizes[1], batch_first=True, bidirectional=True, dropout=0.2)
        self.lstm3 = nn.LSTM(hidden_sizes[1]*2, hidden_sizes[2], batch_first=True, dropout=0.2)
        self.attention = nn.Linear(hidden_sizes[2], 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_sizes[2], hidden_sizes[2]//2)
        self.bn1 = nn.BatchNorm1d(hidden_sizes[2]//2)
        self.fc2 = nn.Linear(hidden_sizes[2]//2, output_size)
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def attention_net(self, lstm_output):
        attn_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context = torch.bmm(attn_weights.transpose(1, 2), lstm_output).squeeze(1)
        return context
    
    def forward(self, x):
        lstm_out, _ = self.lstm1(x)
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out, _ = self.lstm3(lstm_out)
        attn_out = self.attention_net(lstm_out)
        out = self.dropout(attn_out)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        out = self.fc2(out)
        return out

def custom_loss(y_pred, y_true):
    epsilon = 1e-8
    rel_error = torch.abs((y_pred - y_true) / (y_true.abs() + epsilon))
    return torch.mean(rel_error) + 0.5 * nn.MSELoss()(y_pred, y_true)

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=64):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=500, patience=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    def lr_lambda(current_step):
        warmup_steps = 10
        if current_step < warmup_steps:
            return current_step / warmup_steps
        return 1.0
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=20, verbose=True, min_lr=1e-7)
    warmup_scheduler = LambdaLR(optimizer, lr_lambda)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    learning_rates = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if torch.isnan(loss):
                print(f"NaN loss at epoch {epoch+1}")
                return None, None, None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        if epoch < 10:
            warmup_scheduler.step()
        
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
        
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}')
        
        if val_loss < best_val_loss and not np.isnan(val_loss):
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates
    }
    with open('training_metrics.json', 'w') as f:
        json.dump(metrics, f)
    print("Training metrics saved to 'training_metrics.json'")
    
    return train_losses, val_losses, learning_rates

def plot_metrics(train_losses, val_losses, learning_rates):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.close()
    print("Loss plot saved as 'loss_plot.png'")
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, learning_rates, label='Learning Rate', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Over Epochs')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig('lr_plot.png')
    plt.close()
    print("Learning rate plot saved as 'lr_plot.png'")

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
    y_test_actual = np.expm1(y_test_transformed)
    y_pred_actual = np.expm1(np.clip(y_pred_transformed, 0, None))
    
    print("\nEvaluation Results:")
    for i, file_name in enumerate(file_names_test):
        print(f"Schedule: {file_name}")
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
    
    residuals = y_test_actual - y_pred_actual
    print(f"\nDiagnostics:")
    print(f"Mean Residual: {np.mean(residuals):.6f}")
    print(f"Std of Residuals: {np.std(residuals):.6f}")
    
    return y_test_actual, y_pred_actual

def main(main_dir):
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    print(f"Processing directory: {main_dir}")
    train_features, test_features, test_file_names = process_directory(main_dir)
    
    if train_features is None or test_features is None:
        print("Error: Insufficient data to proceed")
        return None
    
    print(f"Total training samples: {len(train_features)}")
    print(f"Total test samples: {len(test_features)}")
    
    if len(train_features) < 50 or len(test_features) == 0:
        print("Error: Insufficient training data for robust model training")
        return None
    
    X_train, y_train, X_test, y_test, y_scaler, input_size = prepare_data_for_model(train_features, test_features, max_seq_len=10)
    
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=64)
    
    model = EnhancedLSTMModel(
        input_size=input_size,
        hidden_sizes=[512, 256, 128],
        output_size=1,
        dropout_rate=0.5
    )
    
    criterion = custom_loss
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    print("Building and training Enhanced LSTM model...")
    train_losses, val_losses, learning_rates = train_model(model, train_loader, test_loader, criterion, optimizer)
    
    if train_losses is None or val_losses is None or learning_rates is None:
        print("Training failed due to NaN losses")
        return None
    
    plot_metrics(train_losses, val_losses, learning_rates)
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(model, X_test, y_test, y_scaler, test_file_names)
    
    return model, y_scaler, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "Tiramisu"
    result = main(main_dir)
    if result is not None:
        model, y_scaler, y_test_actual, y_pred_actual = result
        print("\nEnhanced model training and prediction completed!")
    else:
        print("\nModel training failed!")
