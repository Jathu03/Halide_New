import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def load_dataset(file_path='halide_data.npz'):
    """
    Load the dataset from the .npz file.
    """
    data = np.load(file_path)
    sequences = data['sequences']
    execution_times = data['execution_times']
    return sequences, execution_times

def split_data(sequences, execution_times, test_size=20, val_split=0.2, random_state=42):
    """
    Split data into train, validation, and test sets.
    """
    # Reserve 20 samples for testing
    X_temp, X_test, y_temp, y_test = train_test_split(
        sequences, execution_times, test_size=test_size, random_state=random_state
    )
    
    # Split remaining into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_split, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def build_lstm_model(input_shape):
    """
    Build a simple LSTM model for regression.
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1)  # Single output for execution time
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """
    Train the LSTM model with early stopping.
    """
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    return history

def plot_loss(history, output_file='loss_plot.png'):
    """
    Plot training and validation loss and save to file.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def main():
    # Load dataset
    sequences, execution_times = load_dataset()
    print(f"Loaded dataset with {sequences.shape[0]} samples")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Execution times shape: {execution_times.shape}")
    
    # Standardize execution times
    time_scaler = StandardScaler()
    execution_times = time_scaler.fit_transform(execution_times.reshape(-1, 1)).flatten()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        sequences, execution_times
    )
    print(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}, Test samples: {X_test.shape[0]}")
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    model.summary()
    
    # Train model
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test MAE (standardized): {test_mae:.4f}")
    
    # Inverse transform test predictions and targets for interpretable results
    y_pred = model.predict(X_test, verbose=0)
    y_pred_orig = time_scaler.inverse_transform(y_pred).flatten()
    y_test_orig = time_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Calculate MAE in original scale
    mae_orig = np.mean(np.abs(y_pred_orig - y_test_orig))
    print(f"Test MAE (original scale, ms): {mae_orig:.4f}")
    
    # Plot loss
    plot_loss(history)
    print("Loss plot saved as 'loss_plot.png'")

if __name__ == "__main__":
    main()
