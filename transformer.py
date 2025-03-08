# Weighted average
    ensemble_preds = np.zeros_like(all_preds[0])
    for i, pred in enumerate(all_preds):
        # Reshape if needed
        if pred.shape != ensemble_preds.shape:
            pred = pred.reshape(ensemble_preds.shape)
        ensemble_preds += weights[i] * pred
    
    return ensemble_preds

def evaluate_predictions(y_true, y_pred, scaler=None):
    """Evaluate model predictions with various metrics"""
    # Inverse transform if scaler provided
    if scaler is not None:
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.numpy()
        
        # Reshape if needed
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
            
        y_true = scaler.inverse_transform(y_true)
        y_pred = scaler.inverse_transform(y_pred)
    
    # Convert back to original scale (exp(x) - 1) if log1p was used
    if np.max(y_true) < 100:  # Heuristic to determine if log-transformed
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Percentage errors
    abs_percentage_errors = np.abs((y_true - y_pred) / (y_true + 1e-8)) * 100
    mape = np.mean(abs_percentage_errors)
    median_ape = np.median(abs_percentage_errors)
    
    # R^2 score
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'median_ape': median_ape,
        'r2': r2
    }

def predict_and_evaluate(models, X_test, y_test, X_test_tensor, scaler_y, y_test_actual, test_meta_df, test_file_names, output_dir=None):
    """Get predictions from all models and evaluate them"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_predictions = {}
    all_metrics = {}
    
    # Neural Network predictions
    for name, model in models.items():
        if isinstance(model, nn.Module):
            model.eval()
            model.to(device)
            with torch.no_grad():
                X_input = X_test_tensor.to(device)
                y_pred = model(X_input).cpu()
                
                # Transform back to original scale
                y_pred_transformed = scaler_y.inverse_transform(y_pred.numpy())
                y_pred_final = np.expm1(y_pred_transformed)
                
                all_predictions[name] = y_pred_final.squeeze()
        else:  # Random Forest or other sklearn model
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
                # Transform back to original scale
                y_pred = y_pred.reshape(-1, 1)
                y_pred_transformed = scaler_y.inverse_transform(y_pred)
                y_pred_final = np.expm1(y_pred_transformed)
                
                all_predictions[name] = y_pred_final.squeeze()
    
    # Ensemble prediction (weighted average)
    neural_models = [model for name, model in models.items() if isinstance(model, nn.Module)]
    if len(neural_models) > 1:
        ensemble_weights = [0.4, 0.4, 0.2]  # Example weights for ensemble
        ensemble_pred_scaled = ensemble_predictions(neural_models, X_test_tensor, weights=ensemble_weights)
        ensemble_pred_transformed = scaler_y.inverse_transform(ensemble_pred_scaled)
        ensemble_pred_final = np.expm1(ensemble_pred_transformed)
        all_predictions['ensemble'] = ensemble_pred_final.squeeze()
    
    # Evaluate all predictions
    for name, predictions in all_predictions.items():
        metrics = evaluate_predictions(y_test_actual, predictions)
        all_metrics[name] = metrics
        
        print(f"\n{name} Performance Metrics:")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        print(f"Median APE: {metrics['median_ape']:.2f}%")
        print(f"R²: {metrics['r2']:.4f}")
    
    # Create detailed results dataframe
    results_df = pd.DataFrame({
        'file_name': test_file_names,
        'actual_time': y_test_actual
    })
    
    for name, predictions in all_predictions.items():
        results_df[f'{name}_predicted'] = predictions
        results_df[f'{name}_error_pct'] = ((predictions - y_test_actual) / y_test_actual) * 100
    
    # Add metadata columns if available
    for col in test_meta_df.columns:
        if col not in results_df.columns and col not in ['execution_time', 'execution_time_log']:
            results_df[col] = test_meta_df[col].values
    
    # Sort by error magnitude for the best model
    best_model = min(all_metrics.items(), key=lambda x: x[1]['mape'])[0]
    results_df = results_df.sort_values(by=f'{best_model}_error_pct', key=abs)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_df.to_csv(os.path.join(output_dir, 'prediction_results.csv'), index=False)
    
    return all_predictions, all_metrics, results_df

def plot_results(all_predictions, y_test_actual, test_file_names, output_dir=None):
    """Plot prediction results and error distributions"""
    # 1. Actual vs Predicted plot
    plt.figure(figsize=(12, 8))
    
    # Sort by actual execution time
    sort_idx = np.argsort(y_test_actual)
    x = np.arange(len(y_test_actual))
    
    plt.plot(x, y_test_actual[sort_idx], 'o-', label='Actual', linewidth=2, markersize=8)
    
    for name, predictions in all_predictions.items():
        plt.plot(x, predictions[sort_idx], 'o-', label=f'{name} Predicted', alpha=0.7, markersize=6)
    
    plt.xlabel('Samples (sorted by actual time)', fontsize=12)
    plt.ylabel('Execution Time', fontsize=12)
    plt.title('Actual vs Predicted Execution Times', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'), dpi=300, bbox_inches='tight')
    
    # 2. Error distribution plot
    plt.figure(figsize=(12, 8))
    
    for name, predictions in all_predictions.items():
        percentage_errors = ((predictions - y_test_actual) / y_test_actual) * 100
        plt.hist(percentage_errors, bins=20, alpha=0.6, label=f'{name} (Mean: {np.mean(percentage_errors):.2f}%)')
    
    plt.xlabel('Percentage Error', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Error Distribution', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
    
    # 3. Scatter plot of actual vs predicted
    best_model = None
    min_mape = float('inf')
    
    for name, predictions in all_predictions.items():
        errors = np.abs((predictions - y_test_actual) / y_test_actual) * 100
        mape = np.mean(errors)
        if mape < min_mape:
            min_mape = mape
            best_model = name
    
    if best_model:
        plt.figure(figsize=(10, 10))
        predictions = all_predictions[best_model]
        
        # Create scatter plot
        plt.scatter(y_test_actual, predictions, alpha=0.7)
        
        # Add perfect prediction line
        max_val = max(np.max(y_test_actual), np.max(predictions))
        min_val = min(np.min(y_test_actual), np.min(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        # Add 20% error bounds
        plt.plot([min_val, max_val], [min_val*0.8, max_val*0.8], 'g--', alpha=0.5, label='-20% Bound')
        plt.plot([min_val, max_val], [min_val*1.2, max_val*1.2], 'g--', alpha=0.5, label='+20% Bound')
        
        plt.xlabel('Actual Execution Time', fontsize=12)
        plt.ylabel('Predicted Execution Time', fontsize=12)
        plt.title(f'Actual vs Predicted ({best_model})', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'scatter_plot.png'), dpi=300, bbox_inches='tight')
    
    plt.close('all')

def run_analysis(data_dir, output_dir='results', test_size=10, random_split=True):
    """Main function to run the entire analysis pipeline"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Process data directory
    train_features, test_features, test_file_names, function_metadata = process_directory(
        data_dir, test_size=test_size, random_split=random_split
    )
    
    if train_features is None or test_features is None:
        print("Error in data processing. Exiting.")
        return
    
    # Prepare data for modeling
    (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
     X_train_flat, X_test_flat, scaler_y, input_size,
     y_train_actual, y_test_actual, train_meta_df, test_meta_df, feature_names) = prepare_data_for_model(
        train_features, test_features
    )
    
    # Initialize models
    batch_size = min(16, len(train_features))
    train_loader, test_loader = create_data_loaders(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, batch_size)
    
    # Define models
    hybrid_model = HybridAttentionModel(input_size=input_size)
    mlp_model = MLPModel(input_size=input_size)
    
    # Train models
    print("\nTraining Hybrid Model...")
    hybrid_criterion = custom_loss
    hybrid_optimizer = optim.Adam(hybrid_model.parameters(), lr=0.001, weight_decay=1e-5)
    hybrid_losses = train_model(hybrid_model, train_loader, test_loader, hybrid_criterion, hybrid_optimizer, 
                               num_epochs=300, patience=40, model_name="Hybrid")
    
    print("\nTraining MLP Model...")
    mlp_criterion = custom_loss
    mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=0.001, weight_decay=1e-5)
    mlp_losses = train_model(mlp_model, train_loader, test_loader, mlp_criterion, mlp_optimizer, 
                             num_epochs=300, patience=40, model_name="MLP")
    
    # Train Random Forest
    rf_model, rf_preds = train_random_forest(X_train_flat.numpy(), y_train_tensor.numpy(), 
                                             X_test_flat.numpy(), y_test_tensor.numpy(), feature_names)
    
    # Combine all models
    models = {
        'hybrid': hybrid_model,
        'mlp': mlp_model,
        'random_forest': rf_model
    }
    
    # Predict and evaluate
    predictions, metrics, results_df = predict_and_evaluate(
        models, X_test_flat.numpy(), y_test_tensor.numpy(), X_test_tensor, 
        scaler_y, y_test_actual, test_meta_df, test_file_names, output_dir
    )
    
    # Plot results
    plot_results(predictions, y_test_actual, test_file_names, output_dir)
    
    # Save learning curves
    if hybrid_losses and mlp_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(hybrid_losses[0], label='Hybrid - Train')
        plt.plot(hybrid_losses[1], label='Hybrid - Validation')
        plt.plot(mlp_losses[0], label='MLP - Train')
        plt.plot(mlp_losses[1], label='MLP - Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
    
    # Print final results summary
    print("\nFinal Results Summary:")
    for name, model_metrics in metrics.items():
        print(f"{name}: MAPE={model_metrics['mape']:.2f}%, R²={model_metrics['r2']:.4f}")
    
    return models, predictions, metrics, results_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Prediction Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing JSON data files")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for results")
    parser.add_argument("--test_size", type=int, default=10, help="Number of samples to use for testing")
    parser.add_argument("--random_split", action="store_true", help="Use random split instead of taking last N samples")
    
    args = parser.parse_args()
    
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Test size: {args.test_size}")
    print(f"Random split: {args.random_split}")
    
    models, predictions, metrics, results_df = run_analysis(
        args.data_dir, 
        args.output_dir, 
        args.test_size, 
        args.random_split
    )
    
    print("Analysis completed successfully!")
