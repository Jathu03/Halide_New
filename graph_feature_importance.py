import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats
from pathlib import Path
import warnings
import random
import pickle
import argparse
warnings.filterwarnings('ignore')

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

def process_all_files(main_dir):
    all_features = []
    file_paths = []
    
    main_dir_path = Path(main_dir)
    if not main_dir_path.exists():
        print(f"Directory '{main_dir}' does not exist.")
        return all_features, file_paths
    
    for file_path in main_dir_path.rglob('converted_function_graph.json'):
        print(f"Processing {file_path}...", end='\r')
        features = extract_features_from_file(file_path)
        if features is not None:
            all_features.append(features)
            file_paths.append(str(file_path.relative_to(main_dir_path)))
    
    print(f"Processed {len(all_features)} files successfully.           ")
    return all_features, file_paths

def create_additional_features(df):
    df['log_execution_time'] = np.log1p(df['execution_time'])
    
    if 'sched_points_computed_total' in df.columns:
        df['computation_efficiency'] = df['sched_points_computed_total'] / (df['execution_time'] + 1e-8)
    
    if 'total_bytes_at_production' in df.columns:
        df['bytes_processing_rate'] = df['total_bytes_at_production'] / (df['execution_time'] + 1e-8)
    
    if 'sched_working_set' in df.columns and 'sched_bytes_at_production' in df.columns:
        df['memory_utilization_ratio'] = df['sched_working_set'] / (df['sched_bytes_at_production'] + 1e-8)
    
    return df

def analyze_feature_importance(features_list, output_dir):
    df = pd.DataFrame(features_list)
    df = create_additional_features(df)
    
    print(f"\nExecution time statistics (in seconds):")
    print(f"Min: {df['execution_time'].min():.5f} s")
    print(f"Max: {df['execution_time'].max():.5f} s")
    print(f"Mean: {df['execution_time'].mean():.5f} s")
    print(f"Median: {df['execution_time'].median():.5f} s")
    print(f"Std Dev: {df['execution_time'].std():.5f} s")
    
    y = df['execution_time']
    X = df.drop(['execution_time', 'log_execution_time'] if 'log_execution_time' in df.columns else ['execution_time'], axis=1)
    
    X = X.fillna(0)
    constant_features = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_features:
        print(f"\nRemoving {len(constant_features)} constant features")
        X = X.drop(constant_features, axis=1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Evaluation:")
    print(f"Mean Squared Error: {mse:.5f}")
    print(f"R² Score: {r2:.4f}")
    
    cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='r2')
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f}")
    
    feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
    feature_importances = feature_importances.sort_values(ascending=False)
    
    pearson_correlations = {}
    spearman_correlations = {}
    
    for column in X.columns:
        p_corr, _ = stats.pearsonr(X[column], y)
        s_corr, _ = stats.spearmanr(X[column], y)
        pearson_correlations[column] = p_corr
        spearman_correlations[column] = s_corr
    
    pearson_correlations = pd.Series(pearson_correlations).sort_values(ascending=False, key=abs)
    spearman_correlations = pd.Series(spearman_correlations).sort_values(ascending=False, key=abs)
    
    # Save the model, scaler, and feature names as .pkl files
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/random_forest_model.pkl", 'wb') as f:
        pickle.dump(rf, f)
    with open(f"{output_dir}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    with open(f"{output_dir}/feature_names.pkl", 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    print(f"Model saved to {output_dir}/random_forest_model.pkl")
    print(f"Scaler saved to {output_dir}/scaler.pkl")
    print(f"Feature names saved to {output_dir}/feature_names.pkl")
    
    return feature_importances, pearson_correlations, spearman_correlations, y, df, X, rf, scaler, X.columns

def predict_execution_times(model, scaler, file_paths, main_dir, feature_names):
    predictions = []
    main_dir_path = Path(main_dir)
    
    print("\nPredicting execution times for selected files (in seconds):")
    for file_path in file_paths:
        full_path = main_dir_path / file_path
        if not full_path.exists():
            print(f"File not found: {full_path}")
            continue
        
        features = extract_features_from_file(full_path)
        if features is None:
            print(f"Failed to extract features from {full_path}")
            continue
        
        actual_time = features['execution_time']
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
        
        error = abs(actual_time - predicted_time)
        error_percentage = (error / actual_time) * 100 if actual_time != 0 else float('inf')
        
        predictions.append({
            'file': str(file_path),
            'actual_time_s': actual_time,
            'predicted_time_s': predicted_time,
            'error_percentage': error_percentage
        })
        
        print(f"File: {file_path}")
        print(f"Actual Time: {actual_time:.5f} s")
        print(f"Predicted Time: {predicted_time:.5f} s")
        print(f"Error Percentage: {error_percentage:.2f}%")
        print()
    
    if predictions:
        mean_error_percentage = np.mean([p['error_percentage'] for p in predictions if p['error_percentage'] != float('inf')])
        print(f"Mean Absolute Percentage Error (MAPE): {mean_error_percentage:.2f}%")
    
    return predictions

def predict_custom_input(model, scaler, feature_names, input_file=None, input_features=None):
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

def main(main_dir="Graph_Output", output_dir="analysis_results", custom_input_file=None, custom_input_features=None):
    print(f"Processing files in {main_dir}...")
    features_list, file_paths = process_all_files(main_dir)
    
    if not features_list:
        print("No valid data found in the files.")
        return
    
    print(f"Extracted features from {len(features_list)} files.")
    
    print("Analyzing feature importance...")
    feature_importances, pearson_correlations, spearman_correlations, execution_times, df, X, rf, scaler, feature_names = analyze_feature_importance(features_list, output_dir)
    
    # Predict for 5 random files
    if len(file_paths) >= 5:
        selected_files = random.sample(file_paths, 5)
    else:
        selected_files = file_paths
        print(f"Only {len(file_paths)} files available, using all for prediction.")
    
    predictions = predict_execution_times(rf, scaler, selected_files, main_dir, feature_names)
    
    # Save prediction results to a text file
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/prediction_results.txt", 'w') as f:
        f.write("Execution Time Predictions (in seconds)\n")
        f.write("=====================================\n\n")
        for pred in predictions:
            f.write(f"File: {pred['file']}\n")
            f.write(f"Actual Time: {pred['actual_time_s']:.5f} s\n")
            f.write(f"Predicted Time: {pred['predicted_time_s']:.5f} s\n")
            f.write(f"Error Percentage: {pred['error_percentage']:.2f}%\n\n")
        if predictions:
            mean_mape = np.mean([p['error_percentage'] for p in predictions if p['error_percentage'] != float('inf')])
            f.write(f"Mean Absolute Percentage Error (MAPE): {mean_mape:.2f}%\n")
    
    print(f"Prediction results saved to {output_dir}/prediction_results.txt")
    
    # Handle custom input prediction
    if custom_input_file or custom_input_features:
        print("\nPredicting for custom input (in seconds)...")
        try:
            result = predict_custom_input(rf, scaler, feature_names, input_file=custom_input_file, input_features=custom_input_features)
            print(f"Predicted Execution Time: {result['predicted_time_s']:.5f} s")
            if 'actual_time_s' in result:
                print(f"Actual Execution Time: {result['actual_time_s']:.5f} s")
                print(f"Error Percentage: {result['error_percentage']:.2f}%")
            
            with open(f"{output_dir}/prediction_results.txt", 'a') as f:
                f.write("\nCustom Input Prediction (in seconds)\n")
                f.write("==================================\n")
                f.write(f"Input: {custom_input_file if custom_input_file else 'Feature Dictionary'}\n")
                f.write(f"Predicted Execution Time: {result['predicted_time_s']:.5f} s\n")
                if 'actual_time_s' in result:
                    f.write(f"Actual Execution Time: {result['actual_time_s']:.5f} s\n")
                    f.write(f"Error Percentage: {result['error_percentage']:.2f}%\n")
        
        except Exception as e:
            print(f"Error predicting custom input: {str(e)}")
    
    print("Analysis and predictions complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze feature importance and predict execution times.")
    parser.add_argument('--input-dir', default='Graph_Output', help='Input directory containing converted_function_graph.json files')
    parser.add_argument('--output-dir', default='analysis_results', help='Output directory for predictions and models')
    parser.add_argument('--custom-input-file', help='Path to a custom converted_function_graph.json file for prediction')
    args = parser.parse_args()
    
    main(main_dir=args.input_dir, output_dir=args.output_dir, custom_input_file=args.custom_input_file)
