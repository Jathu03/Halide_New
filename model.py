import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import random

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

def load_dataset(file_path='halide_data.npz'):
    """
    Load the dataset created by the preprocessing script
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file {file_path} not found. Please run the preprocessing script first.")
    
    data = np.load(file_path)
    sequences = data['sequences']
    execution_times = data['execution_times']
    
    print(f"Loaded dataset with {len(sequences)} samples")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Execution times shape: {execution_times.shape}")
    
    return sequences, execution_times

def prepare_train_val_test_split(sequences, execution_times, test_size=20):
    """
    Split the dataset into training, validation and test sets.
    Keep exactly 20 samples for testing as specified.
    """
    # Get total number of samples
    n_samples = len(sequences)
    
    if n_samples <= test_size:
        raise ValueError(f"Not enough samples ({n_samples}) to create a test set of {test_size} samples")
    
    # Create indices and shuffle them
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Select test indices (exactly 20 samples)
    test_indices = indices[:test_size]
    remaining_indices = indices[test_size:]
    
    # Split the remaining data into training and validation (80/20 split)
    train_indices, val_indices = train_test_split(remaining_indices, test_size=0.2, random_state=42)
    
    # Create the splits
    X_train = sequences[train_indices]
    y_train = execution_times[train_indices]
    
    X_val = sequences[val_indices]
    y_val = execution_times[val_indices]
    
    X_test = sequences[test_indices]
    y_test = execution_times[test_indices]
    
    print(f"Split dataset into:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def build_lstm_model(input_shape):
    """
    Build an LSTM model for execution time prediction.
    """
    model = Sequential([
        # First LSTM layer with return sequences to stack another LSTM
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        
        # Second LSTM layer
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        
        # Dense layers for regression
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(1)  # Output layer for execution time prediction
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    
    model.summary()
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """
    Train the LSTM model with early stopping and model checkpointing.
    """
    # Create callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        'best_lstm_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set and calculate error percentages.
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate absolute errors
    absolute_errors = np.abs(y_pred.flatten() - y_test)
    
    # Calculate percentage errors
    percentage_errors = (absolute_errors / np.clip(np.abs(y_test), 1e-10, None)) * 100
    
    # Calculate mean and standard deviation of errors
    mean_absolute_error = np.mean(absolute_errors)
    mean_percentage_error = np.mean(percentage_errors)
    
    print("\nTest Set Evaluation:")
    print(f"Mean Absolute Error: {mean_absolute_error:.4f}")
    print(f"Mean Percentage Error: {mean_percentage_error:.2f}%")
    
    # Print individual predictions and errors
    print("\nIndividual Test Sample Results:")
    print("Sample | Actual Time | Predicted Time | Abs Error | Error %")
    print("-" * 70)
    
    for i in range(len(y_test)):
        print(f"{i:6d} | {y_test[i]:11.2f} | {y_pred[i][0]:14.2f} | {absolute_errors[i]:9.2f} | {percentage_errors[i]:7.2f}%")
    
    return y_pred, absolute_errors, percentage_errors

def plot_results(history, y_test, y_pred):
    """
    Plot training history and prediction results.
    """
    # Create a figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot training & validation loss
    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.set_title('Model Loss')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper right')
    ax1.grid(True)
    
    # Plot training & validation MAE
    ax2.plot(history.history['mean_absolute_error'])
    ax2.plot(history.history['val_mean_absolute_error'])
    ax2.set_title('Mean Absolute Error')
    ax2.set_ylabel('MAE')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper right')
    ax2.grid(True)
    
    # Plot actual vs predicted values
    ax3.scatter(y_test, y_pred)
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax3.set_title('Actual vs Predicted Execution Times')
    ax3.set_xlabel('Actual Time (ms)')
    ax3.set_ylabel('Predicted Time (ms)')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('lstm_results.png')
    plt.show()

def main():
    # Load the dataset
    sequences, execution_times = load_dataset()
    
    # Split data into training, validation, and test sets
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_train_val_test_split(
        sequences, execution_times, test_size=20
    )
    
    # Get the input shape for the model
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"Input shape: {input_shape}")
    
    # Build the LSTM model
    model = build_lstm_model(input_shape)
    
    # Train the model
    model, history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate the model on the test set
    y_pred, absolute_errors, percentage_errors = evaluate_model(model, X_test, y_test)
    
    # Plot results
    plot_results(history, y_test, y_pred.flatten())
    
    print("\nModel training and evaluation complete!")

if __name__ == "__main__":
    main()
