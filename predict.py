import json
import pandas as pd
import pickle
import numpy as np
from pathlib import Path

def extract_features_from_file(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        without_extern = data.get('without_extern', {})
        global_features = without_extern.get('global_features', {})
        
        execution_time_ms = global_features.get('execution_time_ms', None)
        if execution_time_ms is None:
            print(f"No execution time found in {file_path}")
            return None
        
        # Convert execution time from milliseconds to seconds
        execution_time = float(execution_time_ms) / 1000.0
        
        nodes = without_extern.get('nodes', [])
        edges = without_extern.get('edges', [])
        
        features = {
            'execution_time': execution_time,
            'nodes_count': len(nodes),
            'edges_count': len(edges),
            'cache_hits': global_features.get('cache_hits', 0),
            'cache_misses': global_features.get('cache_misses', 0),
            'total_bytes_at_production': 0.0,
            'total_vectors': 0.0,
            'total_parallelism': 0.0
        }
        
        features['node_edge_ratio'] = features['nodes_count'] / (features['edges_count'] + 1e-8)
        
        op_counts = {}
        memory_patterns = {'Broadcast': 0, 'Pointwise': 0, 'Slice': 0, 'Transpose': 0}
        
        for node in nodes:
            stages = node.get('stages', [])
            for stage in stages:
                pipeline_features = stage.get('pipeline_features', {})
                op_hist = pipeline_features.get('op_histogram', {}).get('Float', {})
                for op, count in op_hist.items():
                    op_counts[f'op_{op.lower()}'] = op_counts.get(f'op_{op.lower()}', 0) + count
                
                mem_access = pipeline_features.get('memory_access_patterns', {}).get('Float', {})
                for pattern, values in mem_access.items():
                    memory_patterns[pattern] = memory_patterns.get(pattern, 0) + sum(values)
        
        features.update(op_counts)
        for pattern, value in memory_patterns.items():
            features[f'mem_{pattern.lower()}'] = value
        
        scheduling_features = []
        for node in nodes:
            stages = node.get('stages', [])
            for stage in stages:
                sched = stage.get('schedule_features', {})
                scheduling_features.append(sched)
        
        features['scheduling_count'] = len(scheduling_features)
        
        if scheduling_features:
            important_metrics = [
                'bytes_at_production', 'bytes_at_realization', 'bytes_at_root', 'bytes_at_task',
                'inner_parallelism', 'outer_parallelism', 'num_productions', 'num_realizations',
                'num_scalars', 'num_vectors', 'points_computed_total', 'working_set'
            ]
            
            for metric in important_metrics:
                features[f'sched_{metric}'] = sum(sf.get(metric, 0) for sf in scheduling_features)
            
            features['total_bytes_at_production'] = features['sched_bytes_at_production']
            features['total_vectors'] = features['sched_num_vectors']
            features['total_parallelism'] = sum(sf.get('inner_parallelism', 0) * sf.get('outer_parallelism', 1) 
                                              for sf in scheduling_features)
            
            features['bytes_per_vector'] = (features['total_bytes_at_production'] / 
                                          (features['total_vectors'] + 1e-8))
            features['memory_pressure'] = (features['sched_working_set'] / 
                                         (features['sched_bytes_at_production'] + 1e-8))
            features['bytes_per_parallelism'] = (features['total_bytes_at_production'] / 
                                               (features['total_parallelism'] + 1e-8))
            features['nodes_per_schedule'] = (features['nodes_count'] / 
                                            (features['scheduling_count'] + 1e-8))
        
        op_types = sum(1 for k in op_counts.keys())
        features['avg_ops_per_node'] = sum(op_counts.values()) / (features['nodes_count'] + 1e-8)
        features['op_diversity'] = op_types / (features['nodes_count'] + 1e-8)
        
        return features
    
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

def create_additional_features(df):
    df['log_execution_time'] = np.log1p(df['execution_time'])
    
    if 'sched_points_computed_total' in df.columns:
        df['computation_efficiency'] = df['sched_points_computed_total'] / (df['execution_time'] + 1e-8)
    
    if 'total_bytes_at_production' in df.columns:
        df['bytes_processing_rate'] = df['total_bytes_at_production'] / (df['execution_time'] + 1e-8)
    
    if 'sched_working_set' in df.columns and 'sched_bytes_at_production' in df.columns:
        df['memory_utilization_ratio'] = df['sched_working_set'] / (df['sched_bytes_at_production'] + 1e-8)
    
    return df

def predict_execution_time(model, scaler, feature_names, input_file=None, input_features=None):
    if input_file is None and input_features is None:
        raise ValueError("Either input_file or input_features must be provided")
    
    if input_file:
        features = extract_features_from_file(input_file)
        if features is None:
            raise ValueError(f"Failed to extract features from {input_file}")
    else:
        features = input_features.copy()
    
    actual_time = features.get('execution_time', None)
    feature_df = pd.DataFrame([features])
    feature_df = create_additional_features(feature_df)
    
    X = feature_df.drop(['execution_time', 'log_execution_time'] if 'log_execution_time' in feature_df.columns else ['execution_time'], axis=1)
    X = X.fillna(0)
    
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]
    
    X_scaled = scaler.transform(X)
    predicted_time = model.predict(X_scaled)[0]
    
    result = {
        'predicted_time_s': predicted_time
    }
    
    if actual_time is not None:
        error = abs(actual_time - predicted_time)
        error_percentage = (error / actual_time) * 100 if actual_time != 0 else float('inf')
        result['actual_time_s'] = actual_time
        result['error_percentage'] = error_percentage
    
    return result

def main():
    # Load the saved model, scaler, and feature names
    with open('analysis_results/random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('analysis_results/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('analysis_results/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    # Example 1: Predict using a JSON file
    input_file = 'path/to/your/converted_function_graph.json'  # Replace with actual path
    try:
        result = predict_execution_time(model, scaler, feature_names, input_file=input_file)
        print(f"File: {input_file}")
        print(f"Predicted Execution Time: {result['predicted_time_s']:.5f} s")
        if 'actual_time_s' in result:
            print(f"Actual Execution Time: {result['actual_time_s']:.5f} s")
            print(f"Error Percentage: {result['error_percentage']:.2f}%")
    except Exception as e:
        print(f"Error predicting for file: {str(e)}")
    
    # Example 2: Predict using a feature dictionary
    custom_features = {
        'nodes_count': 10,
        'edges_count': 15,
        'cache_hits': 100,
        'cache_misses': 20,
        'sched_bytes_at_production': 1000000,
        'sched_num_vectors': 50,
        'sched_inner_parallelism': 4,
        'sched_outer_parallelism': 2,
        # Add other features as needed, matching feature_names
    }
    try:
        result = predict_execution_time(model, scaler, feature_names, input_features=custom_features)
        print("\nCustom Features Prediction:")
        print(f"Predicted Execution Time: {result['predicted_time_s']:.5f} s")
    except Exception as e:
        print(f"Error predicting for custom features: {str(e)}")

if __name__ == "__main__":
    main()
