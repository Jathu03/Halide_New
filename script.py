import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and preprocess the dataset
def load_and_preprocess_data(file_path="halide_execution_dataset.csv"):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Separate features and target
    target = df['execution_time_ms']
    features = df.drop(columns=['execution_time_ms', 'file_path'])
    
    # Encode categorical variables
    label_encoders = {}
    for column in ['program_name', 'schedule_name']:
        le = LabelEncoder()
        features[column] = le.fit_transform(features[column])
        label_encoders[column] = le
    
    # Numerical features
    numerical_features = features.drop(columns=['program_name', 'schedule_name'])
    
    # Scale numerical features
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(numerical_features)
    
    # Create sequences with program and schedule IDs
    sequences = []
    for i in range(len(features)):
        seq = [
            features['program_name'].iloc[i],
            features['schedule_name'].iloc[i]
        ] + scaled_numerical[i].tolist()
        sequences.append(seq)
    
    # Pad sequences
    max_seq_length = len(sequences[0])
    X = pad_sequences(sequences, maxlen=max_seq_length, padding='post', dtype='float32')
    
    return X, target.values, label_encoders, scaler, max_seq_length, numerical_features.columns

# Build LSTM model
def build_lstm_model(input_shape, vocab_size=100):
    model = Sequential([
        Input(shape=input_shape),
        Embedding(input_dim=vocab_size, output_dim=32, input_length=input_shape[0]),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Calculate error percentage
def calculate_error_percentage(y_true, y_pred):
    return np.abs((y_true - y_pred) / y_true) * 100

def main():
    try:
        # Load and preprocess data
        X, y, label_encoders, scaler, max_seq_length, num_feature_names = load_and_preprocess_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Build model
        vocab_size = max(
            len(label_encoders['program_name'].classes_),
            len(label_encoders['schedule_name'].classes_)
        ) + 1
        model = build_lstm_model((max_seq_length,), vocab_size)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest MAE: {test_mae:.2f} ms")
        
        # Make predictions
        y_pred = model.predict(X_test).flatten()
        
        # Calculate error percentages for 10 random test samples
        indices = np.random.choice(len(y_test), 10, replace=False)
        sample_errors = []
        
        print("\nError percentages for 10 random test samples:")
        for idx in indices:
            true_time = y_test[idx]
            pred_time = y_pred[idx]
            error_pct = calculate_error_percentage(true_time, pred_time)
            sample_errors.append(error_pct)
            print(f"Sample {idx}:")
            print(f"  True execution time: {true_time:.2f} ms")
            print(f"  Predicted: {pred_time:.2f} ms")
            print(f"  Error percentage: {error_pct:.2f}%")
        
        # Overall statistics
        mean_error_pct = np.mean(sample_errors)
        median_error_pct = np.median(sample_errors)
        print(f"\nMean error percentage: {mean_error_pct:.2f}%")
        print(f"Median error percentage: {median_error_pct:.2f}%")
        
        # Save model
        model.save('halide_execution_time_model.h5')
        print("Model saved to 'halide_execution_time_model.h5'")
        
    except Exception as e:
        print(f"Error in processing: {str(e)}")

if __name__ == "__main__":
    main()
