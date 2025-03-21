import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# Load the precomputed dataset
def load_dataset():
    # Load Halide data
    halide_sequences = np.load('halide_sequences.npy')
    halide_exec_times = np.load('halide_exec_times.npy')
    
    # Load Tiramisu data
    tiramisu_sequences = np.load('tiramisu_sequences.npy')
    tiramisu_exec_times = np.load('tiramisu_exec_times.npy')
    
    # Combine datasets (you can separate them if desired)
    X = np.concatenate([halide_sequences, tiramisu_sequences], axis=0)
    y = np.concatenate([halide_exec_times, tiramisu_exec_times], axis=0)
    
    print("Combined Sequences Shape:", X.shape)  # (n_samples, 50, 44)
    print("Combined Execution Times Shape:", y.shape)  # (n_samples,)
    
    return X, y

# Preprocess data
def preprocess_data(X, y):
    # Normalize execution times
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_scaled, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)
    
    print("Train Shape:", X_train.shape, y_train.shape)
    print("Validation Shape:", X_val.shape, y_val.shape)
    print("Test Shape:", X_test.shape, y_test.shape)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

# Build the model
def build_model(input_shape=(50, 44)):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),  # Embedding layer: learns a dense representation
        Dropout(0.3),
        LSTM(64),  # Reduces sequence to a fixed-size vector
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)  # Predicts normalized execution time
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

# Main execution
def main():
    # Load and preprocess data
    X, y = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess_data(X, y)
    
    # Build and train model
    model = build_model()
    model.summary()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss (MSE): {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Predict and inverse transform for interpretability
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    print("\nSample Predictions (seconds):", y_pred[:5].flatten())
    print("Sample Actuals (seconds):", y_test_original[:5])
    
    # Plot training history
    plot_history(history)
    
    # Save model and scaler
    model.save('execution_predictor.h5')
    np.save('scaler_mean.npy', scaler.mean_)
    np.save('scaler_scale.npy', scaler.scale_)
    print("\nModel and scaler saved.")

if __name__ == "__main__":
    # Ensure data files exist
    required_files = ['halide_sequences.npy', 'halide_exec_times.npy', 
                      'tiramisu_sequences.npy', 'tiramisu_exec_times.npy']
    for file in required_files:
        if not os.path.exists(file):
            print(f"Error: {file} not found. Run transformer.py first to generate the dataset.")
            exit(1)
    
    main()
