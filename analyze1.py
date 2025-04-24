import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict
import uuid

# Function to extract features from a single JSON file
def extract_features(json_data):
    features = {}
    
    # Extract global features
    global_node = next((child for child in json_data['children'] if child['name'] == 'Global Features'), None)
    if global_node:
        features['cache_hits'] = global_node.get('cache_hits', 0)
        features['cache_misses'] = global_node.get('cache_misses', 0)
        features['execution_time_ms'] = global_node.get('execution_time_ms', 0)
    
    # Extract op_histogram features (aggregate across all nodes)
    op_histogram = defaultdict(int)
    for node in json_data['children']:
        if 'op_histogram' in node:
            for op, count in node['op_histogram'].items():
                op_histogram[op.lower()] += count
    for op, count in op_histogram.items():
        features[f'op_{op.lower()}'] = count
    
    # Extract memory patterns (aggregate as sum across nodes)
    memory_patterns = defaultdict(lambda: [0, 0, 0, 0])
    for node in json_data['children']:
        if 'memory_patterns' in node:
            for pattern, values in node['memory_patterns'].items():
                memory_patterns[pattern] = [sum(x) for x in zip(memory_patterns[pattern], values)]
    for pattern, values in memory_patterns.items():
        for i, val in enumerate(values):
            features[f'memory_{pattern.lower()}_{i}'] = val
    
    # Extract scheduling features (aggregate as sum or average where appropriate)
    scheduling_keys = [
        'num_realizations', 'num_productions', 'points_computed_total', 'innermost_loop_extent',
        'inner_parallelism', 'outer_parallelism', 'bytes_at_realization', 'bytes_at_production',
        'bytes_at_root', 'unique_bytes_read_per_realization', 'working_set', 'vector_size',
        'num_vectors', 'num_scalars', 'bytes_at_task', 'working_set_at_task', 'working_set_at_production',
        'working_set_at_realization', 'working_set_at_root'
    ]
    scheduling_sums = defaultdict(float)
    node_count = 0
    for node in json_data['children']:
        if 'scheduling' in node:
            node_count += 1
            for key in scheduling_keys:
                scheduling_sums[key] += node['scheduling'].get(key, 0)
    for key in scheduling_keys:
        # Average for parallelism, sum for others
        if key in ['inner_parallelism', 'outer_parallelism'] and node_count > 0:
            features[f'sched_{key}'] = scheduling_sums[key] / node_count
        else:
            features[f'sched_{key}'] = scheduling_sums[key]
    
    # Derived features with division-by-zero protection
    features['total_parallelism'] = features.get('sched_inner_parallelism', 0) + features.get('sched_outer_parallelism', 0)
    features['scheduling_count'] = features.get('sched_num_realizations', 0) + features.get('sched_num_productions', 0)
    features['total_bytes_at_production'] = features.get('sched_bytes_at_production', 0)
    features['total_vectors'] = features.get('sched_num_vectors', 0)
    
    # Safe division: return 0 if denominator is 0
    features['computation_efficiency'] = (features.get('sched_points_computed_total', 0) /
                                         features.get('sched_bytes_at_realization', 1)) if features.get('sched_bytes_at_realization', 0) != 0 else 0
    features['memory_pressure'] = (features.get('sched_working_set', 0) /
                                  features.get('sched_bytes_at_root', 1)) if features.get('sched_bytes_at_root', 0) != 0 else 0
    features['memory_utilization_ratio'] = (features.get('sched_unique_bytes_read_per_realization', 0) /
                                           features.get('sched_bytes_at_task', 1)) if features.get('sched_bytes_at_task', 0) != 0 else 0
    features['bytes_processing_rate'] = (features.get('sched_bytes_at_realization', 0) /
                                        features.get('execution_time_ms', 1)) if features.get('execution_time_ms', 0) != 0 else 0
    features['bytes_per_parallelism'] = (features.get('sched_bytes_at_task', 0) /
                                        features.get('total_parallelism', 1)) if features.get('total_parallelism', 0) != 0 else 0
    features['bytes_per_vector'] = (features.get('sched_bytes_at_realization', 0) /
                                   features.get('sched_num_vectors', 1)) if features.get('sched_num_vectors', 0) != 0 else 0
    
    # Node and edge counts
    nodes_count = len(json_data['children'])
    edges_count = sum(len(node.get('children', [])) for node in json_data['children'])
    features['nodes_count'] = nodes_count
    features['edges_count'] = edges_count
    features['node_edge_ratio'] = nodes_count / (edges_count + 1)
    features['nodes_per_schedule'] = nodes_count / (features.get('scheduling_count', 1)) if features.get('scheduling_count', 0) != 0 else 0
    features['op_diversity'] = len([k for k, v in features.items() if k.startswith('op_') and v > 0])
    
    return features

# Main function to process all JSON files and generate report
def generate_feature_importance_report(tree_output_dir):
    data = []
    file_count = 0
    
    # Recursively find all tree_representation.json files
    for root, dirs, files in os.walk(tree_output_dir):
        if 'tree_representation.json' in files:
            file_path = os.path.join(root, 'tree_representation.json')
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                features = extract_features(json_data)
                data.append(features)
                file_count += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    if not data:
        return "No valid JSON files found in Tree_Output directory."
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df.fillna(0, inplace=True)
    
    # Extract execution time as target
    y = df['execution_time_ms']
    X = df.drop('execution_time_ms', axis=1)
    
    # Summary statistics
    exec_time_stats = {
        'total_files': file_count,
        'min_exec_time': y.min(),
        'max_exec_time': y.max(),
        'mean_exec_time': y.mean(),
        'median_exec_time': y.median(),
        'std_exec_time': y.std()
    }
    
    # Random Forest feature importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    # Pearson and Spearman correlations
    pearson_corrs = {}
    spearman_corrs = {}
    for col in X.columns:
        pearson_corrs[col] = pearsonr(X[col], y)[0]
        spearman_corrs[col] = spearmanr(X[col], y)[0]
    
    pearson_series = pd.Series(pearson_corrs).sort_values(key=abs, ascending=False)
    spearman_series = pd.Series(spearman_corrs).sort_values(key=abs, ascending=False)
    
    # Generate report
    report = f"""
Feature Importance Analysis Report
=================================

Summary Statistics:
Total files processed: {exec_time_stats['total_files']}
Execution time range: {exec_time_stats['min_exec_time']:.2f} ms to {exec_time_stats['max_exec_time']:.2f} ms
Mean execution time: {exec_time_stats['mean_exec_time']:.2f} ms
Median execution time: {exec_time_stats['median_exec_time']:.2f} ms
Standard deviation: {exec_time_stats['std_exec_time']:.2f} ms

Feature Importance (Random Forest):
----------------------------------
"""
    for feature, importance in feature_importance.items():
        report += f"{feature}: {importance:.4f}\n"
    
    report += f"""
Correlation with Execution Time (Pearson):
-----------------------------------------
"""
    for feature, corr in pearson_series.items():
        report += f"{feature}: {corr:.4f}\n"
    
    report += f"""
Correlation with Execution Time (Spearman):
------------------------------------------
"""
    for feature, corr in spearman_series.items():
        report += f"{feature}: {corr:.4f}\n"
    
    # Save report
    output_path = os.path.join(tree_output_dir, 'feature_importance_report.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report

# Example usage (replace with actual directory path)
if __name__ == "__main__":
    tree_output_dir = "Tree_Output"  # Replace with actual path
    report = generate_feature_importance_report(tree_output_dir)
    print(report)
