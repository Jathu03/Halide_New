import json
import numpy as np
import logging
from typing import List, Dict
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_features(sample: Dict) -> List[float]:
    """
    Extract features from a dataset sample for model training.
    Returns a list of numerical features.
    """
    try:
        # Basic features
        node_count = sample['node_count']
        edge_count = sample['edge_count']
        
        # Edge density (avoid division by zero)
        edge_density = edge_count / max(node_count, 1)
        
        # Node details aggregation
        nodes = sample['nodes']
        output_shapes = []
        element_types = set()
        scheduling_features = 0
        
        for node in nodes:
            details = node.get('Details', {})
            # Extract output shape (e.g., [1, 32, 224, 224])
            if 'output_shape' in details:
                shape = details['output_shape']
                if isinstance(shape, list) and all(isinstance(x, (int, float)) for x in shape):
                    output_shapes.append(shape)
            # Count unique element types
            if 'output_element_type' in details:
                element_types.add(details['output_element_type'])
            # Count non-null scheduling features
            if node.get('scheduling_feature') is not None:
                scheduling_features += 1
        
        # Aggregate output shape features
        avg_shape_dims = np.mean([np.prod(shape) for shape in output_shapes]) if output_shapes else 0
        max_shape_dims = np.max([np.prod(shape) for shape in output_shapes]) if output_shapes else 0
        shape_count = len(output_shapes)
        
        # Number of unique element types
        element_type_count = len(element_types)
        
        # Proportion of nodes with scheduling features
        scheduling_feature_ratio = scheduling_features / max(node_count, 1)
        
        # Collect all features
        features = [
            node_count,
            edge_count,
            edge_density,
            avg_shape_dims,
            max_shape_dims,
            shape_count,
            element_type_count,
            scheduling_feature_ratio
        ]
        
        # Replace NaN or inf with 0 for model compatibility
        features = [float(x) if np.isfinite(x) else 0.0 for x in features]
        
        return features
    
    except Exception as e:
        logging.error(f"Error extracting features from {sample['file_path']}: {e}")
        return [0.0] * 8  # Return zero vector if feature extraction fails

def load_dataset(dataset_file: str) -> tuple[List[List[float]], List[float]]:
    """
    Load the dataset and extract features and target (execution_time_ms).
    Returns features (X) and targets (y).
    """
    if not os.path.exists(dataset_file):
        logging.error(f"Dataset file {dataset_file} not found.")
        raise FileNotFoundError(f"Dataset file {dataset_file} not found.")
    
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
    
    X = []
    y = []
    
    for sample in dataset:
        try:
            features = extract_features(sample)
            execution_time = float(sample['execution_time_ms'])
            X.append(features)
            y.append(execution_time)
        except (KeyError, ValueError) as e:
            logging.warning(f"Skipping sample {sample.get('file_path', 'unknown')}: {e}")
            continue
    
    logging.info(f"Loaded {len(X)} valid samples with {len(X[0])} features each.")
    return X, y

def train_model(X: List[List[float]], y: List[float]) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor on the dataset.
    Returns the trained model.
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate on training set
    y_train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    logging.info(f"Training MSE: {train_mse:.4f}, R2: {train_r2:.4f}")
    
    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    logging.info(f"Test MSE: {test_mse:.4f}, R2: {test_r2:.4f}")
    
    # Feature importance
    feature_names = [
        'node_count', 'edge_count', 'edge_density', 'avg_shape_dims',
        'max_shape_dims', 'shape_count', 'element_type_count', 'scheduling_feature_ratio'
    ]
    importances = model.feature_importances_
    for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        logging.info(f"Feature {name}: {importance:.4f}")
    
    return model

def save_model(model: RandomForestRegressor, model_file: str):
    """
    Save the trained model to a file.
    """
    joblib.dump(model, model_file)
    logging.info(f"Model saved to {model_file}")

if __name__ == "__main__":
    dataset_file = "synthetic_dataset.json"
    model_file = "execution_time_model.pkl"
    
    try:
        # Load and preprocess dataset
        X, y = load_dataset(dataset_file)
        
        if not X or not y:
            logging.error("No valid data loaded. Cannot train model.")
            raise ValueError("No valid data loaded.")
        
        # Train model
        model = train_model(X, y)
        
        # Save model
        save_model(model, model_file)
        
    except FileNotFoundError as e:
        logging.error(f"File error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
