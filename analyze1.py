import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

def get_execution_time(file_path):
    try:
        with open(file_path, 'rb') as f:
            raw_content = f.read()
            content = raw_content.decode('utf-8', errors='replace').replace('\0', '')
            data = json.loads(content)
        
        schedules = data.get("scheduling_data", [])
        for item in schedules:
            if isinstance(item, dict) and item.get('name') == 'total_execution_time_ms':
                execution_time = item.get('value')
                if execution_time is not None:
                    return float(execution_time)
        
        if schedules and isinstance(schedules[-1], dict) and "value" in schedules[-1]:
            execution_time = schedules[-1]["value"]
            return float(execution_time)
        
        return None
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def extract_features_from_file(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    
        execution_time = get_execution_time(file_path)
        if execution_time is None:
            return None
    
        nodes_features = []
        edges_features = []
        programming_details = data.get("programming_details", {})
        
        if 'Nodes' in programming_details:
            for node in programming_details['Nodes']:
                node_feature = {}
                node_feature['Name'] = node.get('Name', '')
                if 'Details' in node and 'Op histogram' in node['Details']:
                    op_hist = node['Details']['Op histogram']
                    for op_line in op_hist:
                        parts = op_line.strip().split(':')
                        if len(parts) == 2:
                            op_name = parts[0].strip()
                            op_count = int(parts[1].strip())
                            node_feature[f'op_{op_name.lower()}'] = op_count
                nodes_features.append(node_feature)
        
        if 'Edges' in programming_details:
            for edge in programming_details['Edges']:
                edge_feature = {}
                edge_feature['From'] = edge.get('From', '')
                edge_feature['To'] = edge.get('To', '')
                edge_feature['Name'] = edge.get('Name', '')
                edges_features.append(edge_feature)
    
        scheduling_features = []
        scheduling_data = data.get("scheduling_data", [])
        if not scheduling_data and 'Schedules' in programming_details:
            scheduling_data = programming_details['Schedules']
    
        if scheduling_data:
            for sched in scheduling_data:
                sched_feature = {}
                sched_feature['Name'] = sched.get('Name', '')
                if 'Details' in sched and 'scheduling_feature' in sched['Details']:
                    sf = sched['Details']['scheduling_feature']
                    for key, value in sf.items():
                        sched_feature[key] = value
                scheduling_features.append(sched_feature)
    
        features = {
            'execution_time': execution_time,
            'nodes_count': len(nodes_features),
            'edges_count': len(edges_features),
            'scheduling_count': len(scheduling_features),
            'total_bytes_at_production': 0.0,
            'total_vectors': 0.0,
            'total_parallelism': 0.0
        }
        
        if len(nodes_features) > 0 and len(edges_features) > 0:
            features['node_edge_ratio'] = len(nodes_features) / len(edges_features)
        else:
            features['node_edge_ratio'] = 0
        
        op_counts = {}
        for node in nodes_features:
            for key, value in node.items():
                if key.startswith('op_'):
                    op_counts[key] = op_counts.get(key, 0) + value
        features.update(op_counts)
        
        if scheduling_features and scheduling_features[0]:
            important_metrics = [
                'bytes_at_production', 'bytes_at_realization', 'bytes_at_root', 'bytes_at_task',
                'inner_parallelism', 'outer_parallelism', 'num_productions', 'num_realizations',
                'num_scalars', 'num_vectors', 'points_computed_total', 'working_set'
            ]
            for metric in important_metrics:
                if metric in scheduling_features[0]:
                    features[f'sched_{metric}'] = scheduling_features[0][metric]
            
            total_bytes_at_production = sum(sf.get('bytes_at_production', 0) for sf in scheduling_features if isinstance(sf, dict))
            total_vectors = sum(sf.get('num_vectors', 0) for sf in scheduling_features if isinstance(sf, dict))
            total_parallelism = sum(sf.get('inner_parallelism', 0) * sf.get('outer_parallelism', 1) for sf in scheduling_features if isinstance(sf, dict))
            
            features['total_bytes_at_production'] = total_bytes_at_production
            features['total_vectors'] = total_vectors
            features['total_parallelism'] = total_parallelism
            
            if total_vectors > 0:
                features['bytes_per_vector'] = total_bytes_at_production / total_vectors
            else:
                features['bytes_per_vector'] = 0
            
            if 'working_set' in scheduling_features[0] and 'bytes_at_production' in scheduling_features[0]:
                features['memory_pressure'] = scheduling_features[0]['working_set'] / scheduling_features[0]['bytes_at_production'] if scheduling_features[0]['bytes_at_production'] > 0 else 0
        
        if len(nodes_features) > 0:
            op_types = sum(1 for k in op_counts.keys())
            features['avg_ops_per_node'] = sum(op_counts.values()) / len(nodes_features)
            features['op_diversity'] = op_types / len(nodes_features) if len(nodes_features) > 0 else 0
        
        features['bytes_per_parallelism'] = features['total_bytes_at_production'] / (features['total_parallelism'] + 1e-8)
        features['nodes_per_schedule'] = features['nodes_count'] / (features['scheduling_count'] + 1e-8)
        
        return features
    
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

def process_all_files(main_dir):
    all_features = []
    file_paths = []
    
    for subdir in os.listdir(main_dir):
        subdir_path = os.path.join(main_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        
        for filename in os.listdir(subdir_path):
            if not filename.endswith('.json'):
                continue
            file_path = os.path.join(subdir_path, filename)
            features = extract_features_from_file(file_path)
            if features is not None:
                all_features.append(features)
                file_paths.append(os.path.join(subdir, filename))
    
    return all_features, file_paths

def analyze_feature_importance(features_list):
    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    
    # Separate features and target
    y = df['execution_time']
    X = df.drop('execution_time', axis=1)
    
    # Fill missing values
    X = X.fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    
    # Feature importance from Random Forest
    feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
    feature_importances = feature_importances.sort_values(ascending=False)
    
    # Compute Pearson correlation coefficients
    correlations = {}
    for column in X.columns:
        corr, _ = stats.pearsonr(X[column], y)
        correlations[column] = corr
    
    correlations = pd.Series(correlations).sort_values(ascending=False, key=abs)
    
    return feature_importances, correlations, y

def generate_report(feature_importances, correlations, execution_times, file_paths, output_file='feature_importance_report.txt'):
    with open(output_file, 'w') as f:
        f.write("Feature Importance Analysis Report\n")
        f.write("=================================\n\n")
        
        f.write("Summary Statistics:\n")
        f.write(f"Total files processed: {len(file_paths)}\n")
        f.write(f"Execution time range: {execution_times.min():.2f} ms to {execution_times.max():.2f} ms\n")
        f.write(f"Mean execution time: {execution_times.mean():.2f} ms\n")
        f.write(f"Median execution time: {execution_times.median():.2f} ms\n\n")
        
        f.write("Feature Importance (Random Forest):\n")
        f.write("----------------------------------\n")
        for feature, importance in feature_importances.items():
            f.write(f"{feature}: {importance:.4f}\n")
        f.write("\n")
        
        f.write("Correlation with Execution Time (Pearson):\n")
        f.write("-----------------------------------------\n")
        for feature, corr in correlations.items():
            f.write(f"{feature}: {corr:.4f}\n")
        f.write("\n")
        
        f.write("Files Processed:\n")
        f.write("---------------\n")
        for file_path in file_paths:
            f.write(f"{file_path}\n")
    
    print(f"Report generated as {output_file}")

def main(main_dir="synthetic_data"):
    print(f"Processing files in {main_dir}...")
    features_list, file_paths = process_all_files(main_dir)
    
    if not features_list:
        print("No valid data found in the files.")
        return
    
    print(f"Extracted features from {len(features_list)} files.")
    
    feature_importances, correlations, execution_times = analyze_feature_importance(features_list)
    
    print("Generating report...")
    generate_report(feature_importances, correlations, execution_times, file_paths)
    
    print("Analysis complete.")

if __name__ == "__main__":
    main()
