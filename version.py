import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# The JSON document
document = '''
<JSON_DOCUMENT_FROM_YOUR_INPUT>
'''

# Parse the JSON document
data = json.loads(document)

# Function to extract execution time and scheduling features
def extract_execution_details(data):
    execution_details = {
        "total_execution_time_ms": None,
        "nodes": []
    }
    scheduling = data["programming_details"]["Scheduling"]
    for entry in scheduling:
        if "name" in entry and entry["name"] == "total_execution_time_ms":
            execution_details["total_execution_time_ms"] = entry["value"]
            continue
        if "Details" in entry and "scheduling_feature" in entry["Details"]:
            node_name = entry["Name"]
            features = entry["Details"]["scheduling_feature"]
            node_features = {
                "name": node_name,
                "points_computed_total": features.get("points_computed_total", 0.0),
                "vector_loads_per_vector": features.get("vector_loads_per_vector", 0.0),
                "scalar_loads_per_scalar": features.get("scalar_loads_per_scalar", 0.0),
                "inner_parallelism": features.get("inner_parallelism", 0.0),
                "outer_parallelism": features.get("outer_parallelism", 0.0),
                "working_set": features.get("working_set", 0.0),
                "working_set_at_realization": features.get("working_set_at_realization", 0.0),
                "num_vectors": features.get("num_vectors", 0.0),
                "num_scalars": features.get("num_scalars", 0.0),
                "inlined_calls": features.get("inlined_calls", 0.0),
                "bytes_at_realization": features.get("bytes_at_realization", 0.0),
                "allocation_bytes_read_per_realization": features.get("allocation_bytes_read_per_realization", 0.0)
            }
            execution_details["nodes"].append(node_features)
    return execution_details

# Extract the details
execution_details = extract_execution_details(data)

# Feature keys to include in the model
feature_keys = [
    "points_computed_total", "vector_loads_per_vector", "scalar_loads_per_scalar",
    "inner_parallelism", "outer_parallelism", "working_set", "working_set_at_realization",
    "num_vectors", "num_scalars", "inlined_calls", "bytes_at_realization",
    "allocation_bytes_read_per_realization"
]

# Convert nodes to a feature matrix
def create_feature_matrix(nodes, feature_keys):
    feature_matrix = []
    for node in nodes:
        feature_vector = [node[key] for key in feature_keys]
        feature_matrix.append(feature_vector)
    return np.array(feature_matrix)

# Extract the feature matrix and target
X = create_feature_matrix(execution_details["nodes"], feature_keys)
y = execution_details["total_execution_time_ms"]

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Reshape X for LSTM: [samples, timesteps, features]
timesteps = X_normalized.shape[0]
features = X_normalized.shape[1]
X_lstm = X_normalized.reshape(1, timesteps, features)

# Simulate additional samples by perturbing the data
num_simulated_samples = 100
X_simulated = []
y_simulated = []

for _ in range(num_simulated_samples):
    noise = np.random.uniform(0.9, 1.1, size=X.shape)
    X_perturbed = X * noise
    X_perturbed_normalized = scaler.transform(X_perturbed)
    X_simulated.append(X_perturbed_normalized)
    noise_y = np.random.uniform(0.9, 1.1)
    y_perturbed = y * noise_y
    y_simulated.append(y_perturbed)

X_simulated = np.array(X_simulated)
y_simulated = np.array(y_simulated)

# Split the simulated data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_simulated, y_simulated, test_size=0.2, random_state=42)

# Build the LSTM model
def build_lstm_model(timesteps, features):
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(timesteps, features)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32))
    model.add(Dropout(0.2))
    model.add(Dense(units=16, activation="relu"))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

# Create and train the model
model = build_lstm_model(timesteps, features)
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss (MSE): {loss:.4f}")
print(f"Test MAE: {mae:.4f}")

# Predict on the original data
y_pred = model.predict(X_lstm)
print(f"\nPredicted Execution Time: {y_pred[0][0]:.4f} ms")
print(f"Actual Execution Time: {y:.4f} ms")

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.show()

# Plot training and validation MAE
plt.figure(figsize=(10, 5))
plt.plot(history.history["mae"], label="Training MAE")
plt.plot(history.history["val_mae"], label="Validation MAE")
plt.title("Training and Validation MAE Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()
plt.show()
