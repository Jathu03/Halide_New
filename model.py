import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Load the preprocessed data
def load_preprocessed_data(data_dir="preprocessed_dataset"):
    sequence_data = np.load(os.path.join(data_dir, "sequence_data.npy"))
    execution_times = np.load(os.path.join(data_dir, "execution_times.npy"))
    return sequence_data, execution_times

# 2. Prepare data for LSTM
def prepare_lstm_data(sequence_data, execution_times):
    # Ensure sequence_data has shape (samples, timesteps, features)
    X = sequence_data
    y = execution_times

    # Normalize execution times
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler_y

# 3. Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)  # Single output for execution time
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# 4. Train and evaluate the model
def train_and_evaluate(model, X_train, y_train, X_test, y_test, scaler_y, epochs=50, batch_size=32):
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Evaluate the model
    train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nTraining MAE: {train_mae:.4f}")
    print(f"Testing MAE: {test_mae:.4f}")

    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Inverse transform predictions and true values
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = scaler_y.inverse_transform(y_pred).flatten()

    # Calculate MAE in original scale
    mae_orig = np.mean(np.abs(y_test_orig - y_pred_orig))
    print(f"MAE in original scale (ms): {mae_orig:.4f}")

    return history, y_test_orig, y_pred_orig

# 5. Plot training history and predictions
def plot_results(history, y_test_orig, y_pred_orig):
    # Plot training & validation loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    
    # Plot training & validation MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # Scatter plot of true vs predicted execution times
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_orig, y_pred_orig, alpha=0.5)
    plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--')
    plt.title('True vs Predicted Execution Times')
    plt.xlabel('True Execution Time (ms)')
    plt.ylabel('Predicted Execution Time (ms)')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load data
    data_dir = "preprocessed_dataset"
    sequence_data, execution_times = load_preprocessed_data(data_dir)
    
    print("Sequence Data Shape:", sequence_data.shape)
    print("Execution Times Shape:", execution_times.shape)

    if sequence_data.size == 0 or execution_times.size == 0:
        print("Error: No valid data loaded. Please check the preprocessing step.")
        exit()

    # Prepare data
    X_train, X_test, y_train, y_test, scaler_y = prepare_lstm_data(sequence_data, execution_times)
    
    # Build model
    input_shape = (sequence_data.shape[1], sequence_data.shape[2])  # (timesteps, features)
    model = build_lstm_model(input_shape)
    model.summary()

    # Train and evaluate
    history, y_test_orig, y_pred_orig = train_and_evaluate(
        model, X_train, y_train, X_test, y_test, scaler_y,
        epochs=50, batch_size=32
    )

    # Plot results
    plot_results(history, y_test_orig, y_pred_orig)

    # Save the model
    model.save("lstm_execution_time_model.h5")
    print("Model saved to lstm_execution_time_model.h5")
