import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Attention, LayerNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load the preprocessed dataset
def load_dataset(data_dir="preprocessed_dataset"):
    sequence_data = np.load(f"{data_dir}/sequence_data.npy")  # Shape: (files, timesteps, features)
    edge_df = pd.read_csv(f"{data_dir}/edge_features.csv")
    node_df = pd.read_csv(f"{data_dir}/node_features.csv")
    execution_times = np.load(f"{data_dir}/execution_times.npy")  # Shape: (files,)
    return sequence_data, edge_df, node_df, execution_times

# Define the LSTM with Attention model
def build_lstm_attention_model(input_shape):
    # Input layer for sequences
    inputs = Input(shape=input_shape)  # (timesteps, features)
    
    # LSTM layer
    lstm_out, state_h, state_c = LSTM(64, return_sequences=True, return_state=True)(inputs)
    lstm_out = Dropout(0.2)(lstm_out)
    
    # Attention mechanism
    attention = Attention()([lstm_out, lstm_out])  # Query and value are the same (self-attention)
    attention = LayerNormalization()(attention)
    
    # Combine LSTM output with attention
    context = tf.reduce_mean(attention, axis=1)  # Reduce to (batch_size, 64)
    
    # Dense layers for regression
    dense = Dense(32, activation='relu')(context)
    dense = Dropout(0.2)(dense)
    outputs = Dense(1)(dense)  # Single value for execution time
    
    # Build and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load dataset
    sequence_data, edge_df, node_df, execution_times = load_dataset()
    print("Sequence Data Shape:", sequence_data.shape)
    print("Execution Times Shape:", execution_times.shape)
    print("Edge DataFrame Shape:", edge_df.shape)
    print("Node DataFrame Shape:", node_df.shape)
    
    # Normalize execution times
    scaler = StandardScaler()
    execution_times_scaled = scaler.fit_transform(execution_times.reshape(-1, 1)).flatten()
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        sequence_data, execution_times_scaled, test_size=0.2, random_state=42
    )
    print("Train Shape:", X_train.shape, y_train.shape)
    print("Test Shape:", X_test.shape, y_test.shape)
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
    model = build_lstm_attention_model(input_shape)
    model.summary()
    
    # Train model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f} (scaled)")
    
    # Inverse transform predictions to original scale
    y_pred_train = scaler.inverse_transform(model.predict(X_train)).flatten()
    y_pred_test = scaler.inverse_transform(model.predict(X_test)).flatten()
    y_train_orig = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    train_mae_orig = np.mean(np.abs(y_train_orig - y_pred_train))
    test_mae_orig = np.mean(np.abs(y_test_orig - y_pred_test))
    print(f"Train MAE (original scale): {train_mae_orig:.4f} ms")
    print(f"Test MAE (original scale): {test_mae_orig:.4f} ms")
    
    # Plot training history
    plot_history(history)
    
    # Save the model
    model.save("lstm_attention_model.h5")
    print("Model saved to lstm_attention_model.h5")
    
    # Save the scaler for later use
    np.save("execution_time_scaler.npy", [scaler.mean_, scaler.scale_])
    print("Scaler saved to execution_time_scaler.npy")
