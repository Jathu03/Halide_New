import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_dataset(dataset_path):
    """Load the dataset from the .npz file."""
    data = np.load(dataset_path)
    X = data['sequences']
    y = data['execution_times']
    return X, y

def create_lstm_model(input_shape):
    """Define the LSTM model architecture."""
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)  # Regression output for execution time
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def calculate_mape(actual, predicted):
    """Calculate Mean Absolute Percentage Error for each sample."""
    actual = np.squeeze(actual)
    predicted = np.squeeze(predicted)
    # Avoid division by zero
    mask = actual != 0
    mape = np.abs((actual[mask] - predicted[mask]) / actual[mask]) * 100
    return mape

def train_and_evaluate(dataset_path, test_size=10, epochs=50, batch_size=32):
    """Train the LSTM model and evaluate on test set."""
    # Load dataset
    X, y = load_dataset(dataset_path)
    print(f"Dataset loaded. X shape: {X.shape}, y shape: {y.shape}")

    # Ensure test_size does not exceed number of samples
    if test_size >= len(X):
        print(f"Error: test_size ({test_size}) is larger than dataset size ({len(X)}).")
        return

    # Split into train+val and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Split train+val into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42
    )

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(X_test)}")

    # Create and compile model
    model = create_lstm_model(input_shape=(X.shape[1], X.shape[2]))

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Evaluate on test set
    test_predictions = model.predict(X_test)
    mape_scores = calculate_mape(y_test, test_predictions)

    # Print individual and average MAPE
    print("\nTest Set Results:")
    print("-----------------")
    for i, (actual, pred, mape) in enumerate(zip(y_test.flatten(), test_predictions.flatten(), mape_scores)):
        print(f"Sample {i+1}: Actual = {actual:.2f} ms, Predicted = {pred:.2f} ms, MAPE = {mape:.2f}%")
    avg_mape = np.mean(mape_scores)
    print(f"\nAverage MAPE on test set: {avg_mape:.2f}%")

    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

    # Save model
    model.save('lstm_execution_time_model.h5')
    print("Model saved to lstm_execution_time_model.h5")
    print("Training history plot saved to training_history.png")

if __name__ == "__main__":
    # Define dataset path
    dataset_path = 'lstm_dataset.npz'
    
    # Train and evaluate
    train_and_evaluate(dataset_path, test_size=10, epochs=50, batch_size=32)
