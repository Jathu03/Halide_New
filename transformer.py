import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ... (Previous functions remain unchanged up to ensemble_predictions) ...

def ensemble_predictions(models, X_test, weights=None):
    """Ensemble predictions from multiple models with optional weighting"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_preds = []
    
    for model in models:
        if isinstance(model, torch.nn.Module):
            model.eval()
            model.to(device)
            with torch.no_grad():
                if isinstance(X_test, torch.Tensor):
                    X_test_device = X_test.to(device)
                    preds = model(X_test_device).cpu().numpy()
                else:
                    raise ValueError("X_test must be a tensor for PyTorch models")
        else:  # Assume sklearn model
            preds = model.predict(X_test)
        all_preds.append(preds)
    
    # Apply weights if provided
    if weights is None:
        weights = [1/len(models)] * len(models)
    
    # Ensure weights sum to 1
    weights = np.array(weights) / sum(weights)
    
    # Compute weighted average
    ensemble_preds = np.average(all_preds, axis=0, weights=weights)
    
    return ensemble_preds

def evaluate_model(predictions, y_test, y_scaler, file_names_test, y_test_actual, model_name="Model"):
    """Evaluate predictions and print results"""
    # Convert predictions back to original scale
    if predictions.ndim > 1:
        predictions = predictions.squeeze()
    y_test_scaled = y_test.squeeze()
    
    y_pred_transformed = y_scaler.inverse_transform(predictions.reshape(-1, 1))
    y_test_transformed = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1))
    
    y_pred_actual = np.expm1(np.clip(y_pred_transformed, 0, None))
    y_test_actual_pred = np.expm1(y_test_transformed)
    
    # Use actual execution times from original data for comparison
    y_test_actual = y_test_actual.reshape(-1)
    
    print(f"\n{model_name} Evaluation Results:")
    for i, file_name in enumerate(file_names_test):
        print(f"Schedule: {file_name}")
        print(f"  Actual execution time: {y_test_actual[i]:.6f} seconds")
        print(f"  Predicted execution time: {y_pred_actual[i][0]:.6f} seconds")
        print(f"  Error percentage: {abs(y_test_actual[i] - y_pred_actual[i][0]) / y_test_actual[i] * 100:.2f}%")
    
    mse = mean_squared_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    mape = np.mean(np.abs((y_test_actual - y_pred_actual.flatten()) / (y_test_actual + 1e-8))) * 100
    
    print(f"\n{model_name} Overall Performance:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"MAPE: {mape:.2f}%")
    
    return y_test_actual, y_pred_actual.flatten()

def plot_results(y_test_actual, y_pred_actual, model_name):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_actual, y_pred_actual, alpha=0.5)
    plt.plot([min(y_test_actual), max(y_test_actual)], 
             [min(y_test_actual), max(y_test_actual)], 
             'r--', lw=2)
    plt.xlabel('Actual Execution Time (seconds)')
    plt.ylabel('Predicted Execution Time (seconds)')
    plt.title(f'{model_name}: Actual vs Predicted Execution Times')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_results.png')
    plt.close()

def main(main_dir):
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    print(f"Processing directory: {main_dir}")
    train_features, test_features, test_file_names, function_metadata = process_directory(main_dir)
    
    if train_features is None or test_features is None:
        print("Error: Insufficient data to proceed")
        return None
    
    print(f"Total training samples: {len(train_features)}")
    print(f"Total test samples: {len(test_features)}")
    
    if len(train_features) < 50 or len(test_features) == 0:
        print("Error: Insufficient training data for robust model training")
        return None
    
    (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,
     X_train_flat, X_test_flat, y_scaler, input_size,
     y_train_actual, y_test_actual, train_meta_df, test_meta_df, feature_names) = prepare_data_for_model(train_features, test_features)
    
    # Create data loaders
    train_loader_hybrid, test_loader_hybrid = create_data_loaders(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
    train_loader_mlp, test_loader_mlp = create_data_loaders(X_train_flat, y_train_tensor, X_test_flat, y_test_tensor)
    
    # Initialize models
    hybrid_model = HybridAttentionModel(input_size=input_size)
    mlp_model = MLPModel(input_size=input_size)
    
    # Define optimizers and criterion
    hybrid_optimizer = optim.AdamW(hybrid_model.parameters(), lr=0.001, weight_decay=1e-4)
    mlp_optimizer = optim.AdamW(mlp_model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = custom_loss
    
    # Train models
    hybrid_train_losses, hybrid_val_losses = train_model(
        hybrid_model, train_loader_hybrid, test_loader_hybrid, criterion, hybrid_optimizer,
        model_name="Hybrid Attention Model"
    )
    
    mlp_train_losses, mlp_val_losses = train_model(
        mlp_model, train_loader_mlp, test_loader_mlp, criterion, mlp_optimizer,
        model_name="MLP Model"
    )
    
    # Train Random Forest
    rf_model, rf_preds_scaled = train_random_forest(
        X_train_flat, y_train_tensor, X_test_flat, y_test_tensor, feature_names
    )
    
    # Get individual predictions
    hybrid_preds_scaled = hybrid_model(X_test_tensor).detach().cpu().numpy()
    mlp_preds_scaled = mlp_model(X_test_flat).detach().cpu().numpy()
    
    # Ensemble predictions (equal weights for simplicity)
    ensemble_preds_scaled = ensemble_predictions(
        [hybrid_model, mlp_model, rf_model],
        X_test_flat,  # Use flat format as RF requires it
        weights=[0.4, 0.3, 0.3]  # Weight hybrid higher as it's more complex
    )
    
    # Evaluate all models
    hybrid_y_test_actual, hybrid_y_pred_actual = evaluate_model(
        hybrid_preds_scaled, y_test_tensor.numpy(), y_scaler, test_file_names, y_test_actual,
        "Hybrid Attention Model"
    )
    
    mlp_y_test_actual, mlp_y_pred_actual = evaluate_model(
        mlp_preds_scaled, y_test_tensor.numpy(), y_scaler, test_file_names, y_test_actual,
        "MLP Model"
    )
    
    rf_y_test_actual, rf_y_pred_actual = evaluate_model(
        rf_preds_scaled, y_test_tensor.numpy(), y_scaler, test_file_names, y_test_actual,
        "Random Forest Model"
    )
    
    ensemble_y_test_actual, ensemble_y_pred_actual = evaluate_model(
        ensemble_preds_scaled, y_test_tensor.numpy(), y_scaler, test_file_names, y_test_actual,
        "Ensemble Model"
    )
    
    # Plot results
    plot_results(hybrid_y_test_actual, hybrid_y_pred_actual, "Hybrid Attention Model")
    plot_results(mlp_y_test_actual, mlp_y_pred_actual, "MLP Model")
    plot_results(rf_y_test_actual, rf_y_pred_actual, "Random Forest Model")
    plot_results(ensemble_y_test_actual, ensemble_y_pred_actual, "Ensemble Model")
    
    return {
        'hybrid_model': hybrid_model,
        'mlp_model': mlp_model,
        'rf_model': rf_model,
        'y_scaler': y_scaler,
        'results': {
            'hybrid': (hybrid_y_test_actual, hybrid_y_pred_actual),
            'mlp': (mlp_y_test_actual, mlp_y_pred_actual),
            'rf': (rf_y_test_actual, rf_y_pred_actual),
            'ensemble': (ensemble_y_test_actual, ensemble_y_pred_actual)
        }
    }

if __name__ == "__main__":
    main_dir = "Tiramisu"
    result = main(main_dir)
    if result is not None:
        print("\nModel training and evaluation completed!")
        # Optionally save models
        torch.save(result['hybrid_model'].state_dict(), 'hybrid_model.pth')
        torch.save(result['mlp_model'].state_dict(), 'mlp_model.pth')
        import joblib
        joblib.dump(result['rf_model'], 'rf_model.pkl')
    else:
        print("\nModel training failed!")
