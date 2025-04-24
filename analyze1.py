import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def get_execution_time(file_path):
    try:
        with open(file_path, 'rb') as f:
            raw_content = f.read()
            content = raw_content.decode('utf-8', errors='replace').replace('\0', '')
            data = json.loads(content)
        
        for node in data.get('nodes', []):
            if node.get('name') == 'Global Features':
                execution_time = node.get('execution_time_ms')
                if execution_time is not None:
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
    
        nodes = data.get('nodes', [])
        node_features = []
        dependency_features = []
        
        for node in nodes:
            node_feature = {}
            node_feature['name'] = node.get('name', '')
            node_feature['id'] = node.get('id', -1)
            
            # Extract op_histogram
            op_hist = node.get('op_histogram', {})
            for op_name, op_count in op_hist.items():
                node_feature[f'op_{op_name.lower()}'] = op_count
            
            # Extract memory_patterns
            mem_patterns = node.get('memory_patterns', {})
            for pattern_name, pattern_value in mem_patterns.items():
                node_feature[f'mem_{pattern_name.lower()}'] = pattern_value
            
            # Extract scheduling features
            sched = node.get('scheduling', {})
            for key, value in sched.items():
                node_feature[f'sched_{key}'] = value
            
            # Extract cache metrics for Global Features node
            if node.get('name') == 'Global Features':
                node_feature['cache_hits'] = node.get('cache_hits', 0)
                node_feature['cache_misses'] = node.get('cache_misses', 0)
            
            node_features.append(node_feature)
            
            # Extract dependency features
            for child in node.get('children', []):
                dep_feature = {
                    'from_id': node['id'],
                    'to_id': child.get('target_id', -1),
                    'footprint': child.get('footprint', []),
                    'load_jacobian': child.get('load_jacobian', [])
                }
                dependency_features.append(dep_feature)
    
        features = {
            'execution_time': execution_time,
            'nodes_count': len(node_features),
            'dependencies_count': len(dependency_features),
            'total_bytes_at_realization': 0.0,
            'total_vectors': 0.0,
            'total_parallelism': 0.0
        }
        
        if len(node_features) > 0 and len(dependency_features) > 0:
            features['node_dep_ratio'] = len(node_features) / len(dependency_features)
        else:
            features['node_dep_ratio'] = 0
        
        # Aggregate operation counts
        op_counts = {}
        for node in node_features:
            for key, value in node.items():
                if key.startswith('op_'):
                    op_counts[key] = op_counts.get(key, 0) + value
        features.update(op_counts)
        
        # Aggregate memory patterns
        mem_counts = {}
        for node in node_features:
            for key, value in node.items():
                if key.startswith('mem_'):
                    mem_counts[key] = mem_counts.get(key, 0) + (value if isinstance(value, (int, float)) else sum(value))
        features.update(mem_counts)
        
        # Aggregate scheduling features
        if node_features:
            important_metrics = [
                'bytes_at_realization', 'bytes_at_root', 'bytes_at_task',
                'inner_parallelism', 'outer_parallelism', 'num_realizations',
                'num_vectors', 'points_computed_total', 'working_set_at_task'
            ]
            for metric in important_metrics:
                metric_key = f'sched_{metric}'
                values = [nf.get(metric_key, 0) for nf in node_features]
                features[f'avg_{metric_key}'] = sum(values) / len(node_features) if values else 0
            
            total_bytes_at_realization = sum(nf.get('sched_bytes_at_realization', 0) for nf in node_features)
            total_vectors = sum(nf.get('sched_num_vectors', 0) for nf in node_features)
            total_parallelism = sum(nf.get('sched_inner_parallelism', 0) * nf.get('sched_outer_parallelism', 1) for nf in node_features)
            
            features['total_bytes_at_realization'] = total_bytes_at_realization
            features['total_vectors'] = total_vectors
            features['total_parallelism'] = total_parallelism
            
            if total_vectors > 0:
                features['bytes_per_vector'] = total_bytes_at_realization / total_vectors
            else:
                features['bytes_per_vector'] = 0
            
            working_set = features.get('avg_sched_working_set_at_task', 0)
            bytes_at_real = features.get('avg_sched_bytes_at_realization', 0)
            if bytes_at_real > 0:
                features['memory_pressure'] = working_set / bytes_at_real
            else:
                features['memory_pressure'] = 0
        
        # Global features
        for node in node_features:
            if node['name'] == 'Global Features':
                features['cache_hits'] = node.get('cache_hits', 0)
                features['cache_misses'] = node.get('cache_misses', 0)
                features['cache_hit_ratio'] = features['cache_hits'] / (features['cache_hits'] + features['cache_misses'] + 1e-8)
        
        # Dependency features
        if dependency_features:
            footprint_lengths = [len(dep['footprint']) for dep in dependency_features]
            jacobian_lengths = [len(dep['load_jacobian']) for dep in dependency_features]
            features['avg_footprint_length'] = sum(footprint_lengths) / len(footprint_lengths) if footprint_lengths else 0
            features['avg_jacobian_length'] = sum(jacobian_lengths) / len(jacobian_lengths) if jacobian_lengths else 0
        
        if len(node_features) > 0:
            op_types = sum(1 for k in op_counts.keys())
            features['avg_ops_per_node'] = sum(op_counts.values()) / len(node_features)
            features['op_diversity'] = op_types / len(node_features) if len(node_features) > 0 else 0
        
        features['bytes_per_parallelism'] = features['total_bytes_at_realization'] / (features['total_parallelism'] + 1e-8)
        features['nodes_per_dependency'] = features['nodes_count'] / (features['dependencies_count'] + 1e-8)
        
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
    
    for subdir_path in main_dir_path.iterdir():
        if not subdir_path.is_dir():
            continue
        
        for file_path in subdir_path.glob('tree_representation.json'):
            print(f"Processing {file_path}...", end='\r')
            features = extract_features_from_file(file_path)
            if features is not None:
                all_features.append(features)
                file_paths.append(f"{subdir_path.name}/{file_path.name}")
    
    print(f"Processed {len(all_features)} files successfully.           ")
    return all_features, file_paths

def create_additional_features(df):
    """Create additional derived features that might help the model"""
    df['log_execution_time'] = np.log1p(df['execution_time'])
    
    if 'avg_sched_points_computed_total' in df.columns and 'execution_time' in df.columns:
        df['computation_efficiency'] = df['avg_sched_points_computed_total'] / (df['execution_time'] + 1e-8)
    
    if 'total_bytes_at_realization' in df.columns and 'execution_time' in df.columns:
        df['bytes_processing_rate'] = df['total_bytes_at_realization'] / (df['execution_time'] + 1e-8)
    
    if 'avg_sched_working_set_at_task' in df.columns and 'avg_sched_bytes_at_realization' in df.columns:
        df['memory_utilization_ratio'] = df['avg_sched_working_set_at_task'] / (df['avg_sched_bytes_at_realization'] + 1e-8)
    
    return df

def analyze_feature_importance(features_list):
    df = pd.DataFrame(features_list)
    df = create_additional_features(df)
    
    print(f"\nExecution time statistics:")
    print(f"Min: {df['execution_time'].min():.2f} ms")
    print(f"Max: {df['execution_time'].max():.2f} ms")
    print(f"Mean: {df['execution_time'].mean():.2f} ms")
    print(f"Median: {df['execution_time'].median():.2f} ms")
    print(f"Std Dev: {df['execution_time'].std():.2f} ms")
    
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
    print(f"Mean Squared Error: {mse:.2f}")
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
    
    return feature_importances, pearson_correlations, spearman_correlations, y, df, X

def plot_feature_importance(feature_importances, correlations, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    top_features = feature_importances.head(15)
    sns.barplot(x=top_features.values, y=top_features.index)
    plt.title('Top 15 Features by Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_features_importance.png")
    
    plt.figure(figsize=(12, 8))
    top_correlations = correlations.head(15)
    sns.barplot(x=top_correlations.values, y=top_correlations.index)
    plt.title('Top 15 Features by Correlation with Execution Time')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_features_correlation.png")
    
    plt.close('all')

def plot_execution_time_distribution(execution_times, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(execution_times, kde=True)
    plt.title('Distribution of Execution Times')
    plt.xlabel('Execution Time (ms)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/execution_time_distribution.png")
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=execution_times)
    plt.title('Execution Time Box Plot')
    plt.xlabel('Execution Time (ms)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/execution_time_boxplot.png")
    
    plt.close('all')

def plot_scatter_for_top_features(df, top_features, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    top_feature_names = list(top_features.index[:10])
    
    for feature in top_feature_names:
        if feature in df.columns:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=df[feature], y=df['execution_time'])
            plt.title(f'{feature} vs Execution Time')
            plt.xlabel(feature)
            plt.ylabel('Execution Time (ms)')
            plt.tight_layout()
            safe_feature = feature.replace('/', '_').replace('\\', '_')
            plt.savefig(f"{output_dir}/scatter_{safe_feature}.png")
            plt.close()

def generate_report(feature_importances, pearson_correlations, spearman_correlations, 
                    execution_times, file_paths, df, output_dir='analysis_results'):
    
    os.makedirs(output_dir, exist_ok=True)
    
    plot_execution_time_distribution(execution_times, output_dir)
    plot_feature_importance(feature_importances, pearson_correlations, output_dir)
    plot_scatter_for_top_features(df, feature_importances, output_dir)
    
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Feature Importance Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .section {{ margin-bottom: 30px; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>Feature Importance Analysis Report</h1>
        
        <div class="section">
            <h2>Summary Statistics</h2>
            <p>Total files processed: {len(file_paths)}</p>
            <p>Execution time range: {execution_times.min():.2f} ms to {execution_times.max():.2f} ms</p>
            <p>Mean execution time: {execution_times.mean():.2f} ms</p>
            <p>Median execution time: {execution_times.median():.2f} ms</p>
            <p>Standard deviation: {execution_times.std():.2f} ms</p>
            
            <h3>Execution Time Distribution</h3>
            <img src="execution_time_distribution.png" alt="Execution Time Distribution">
            <img src="execution_time_boxplot.png" alt="Execution Time Box Plot">
        </div>
        
        <div class="section">
            <h2>Feature Importance Analysis</h2>
            
            <h3>Top Features by Importance (Random Forest Model)</h3>
            <img src="top_features_importance.png" alt="Top Features by Importance">
            
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Feature</th>
                    <th>Importance Score</th>
                </tr>
    """
    
    top_features = list(feature_importances.items())[:20]
    for i, (feature, importance) in enumerate(top_features, start=1):
        html_report += f"""
                <tr>
                    <td>{i}</td>
                    <td>{feature}</td>
                    <td>{importance:.4f}</td>
                </tr>
        """
    
    html_report += """
            </table>
            
            <h3>Top Features by Correlation (Pearson)</h3>
            <img src="top_features_correlation.png" alt="Top Features by Correlation">
            
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Feature</th>
                    <th>Pearson Correlation</th>
                    <th>Spearman Correlation</th>
                </tr>
    """
    
    top_pearson = list(pearson_correlations.items())[:20]
    for i, (feature, corr) in enumerate(top_pearson, start=1):
        spearman_corr = spearman_correlations.get(feature, 0)
        html_report += f"""
                <tr>
                    <td>{i}</td>
                    <td>{feature}</td>
                    <td>{corr:.4f}</td>
                    <td>{spearman_corr:.4f}</td>
                </tr>
        """
    
    html_report += """
            </table>
        </div>
        
        <div class="section">
            <h2>Scatter Plots of Top Features vs Execution Time</h2>
    """
    
    top_feature_names = list(feature_importances.index[:10])
    for feature in top_feature_names:
        safe_feature = feature.replace('/', '_').replace('\\', '_')
        html_report += f"""
            <h3>{feature} vs Execution Time</h3>
            <img src="scatter_{safe_feature}.png" alt="Scatter plot of {feature}">
        """
    
    html_report += """
        </div>
        
        <div class="section">
            <h2>Files Processed</h2>
            <table>
                <tr>
                    <th>#</th>
                    <th>File Path</th>
                </tr>
    """
    
    display_files = file_paths[:100]
    for i, file_path in enumerate(display_files, start=1):
        html_report += f"""
                <tr>
                    <td>{i}</td>
                    <td>{file_path}</td>
                </tr>
        """
    
    if len(file_paths) > 100:
        html_report += f"""
                <tr>
                    <td colspan="2">... and {len(file_paths) - 100} more files</td>
                </tr>
        """
    
    html_report += """
            </table>
        </div>
    </body>
    </html>
    """
    
    with open(f"{output_dir}/feature_importance_report.html", 'w') as f:
        f.write(html_report)
    
    with open(f"{output_dir}/feature_importance_report.txt", 'w') as f:
        f.write("Feature Importance Analysis Report\n")
        f.write("=================================\n\n")
        
        f.write("Summary Statistics:\n")
        f.write(f"Total files processed: {len(file_paths)}\n")
        f.write(f"Execution time range: {execution_times.min():.2f} ms to {execution_times.max():.2f} ms\n")
        f.write(f"Mean execution time: {execution_times.mean():.2f} ms\n")
        f.write(f"Median execution time: {execution_times.median():.2f} ms\n")
        f.write(f"Standard deviation: {execution_times.std():.2f} ms\n\n")
        
        f.write("Feature Importance (Random Forest):\n")
        f.write("----------------------------------\n")
        for feature, importance in list(feature_importances.items())[:30]:
            f.write(f"{feature}: {importance:.4f}\n")
        f.write("\n")
        
        f.write("Correlation with Execution Time (Pearson):\n")
        f.write("-----------------------------------------\n")
        for feature, corr in list(pearson_correlations.items())[:30]:
            f.write(f"{feature}: {corr:.4f}\n")
        f.write("\n")
        
        f.write("Correlation with Execution Time (Spearman):\n")
        f.write("------------------------------------------\n")
        for feature, corr in list(spearman_correlations.items())[:30]:
            f.write(f"{feature}: {corr:.4f}\n")
        f.write("\n")
    
    print(f"Reports generated in the '{output_dir}' directory")
    print(f"- HTML report: {output_dir}/feature_importance_report.html")
    print(f"- Text report: {output_dir}/feature_importance_report.txt")

def main(main_dir="Tree_Output", output_dir="analysis_results"):
    print(f"Processing files in {main_dir}...")
    features_list, file_paths = process_all_files(main_dir)
    
    if not features_list:
        print("No valid data found in the files.")
        return
    
    print(f"Extracted features from {len(features_list)} files.")
    
    print("Analyzing feature importance...")
    feature_importances, pearson_correlations, spearman_correlations, execution_times, df, X = analyze_feature_importance(features_list)
    
    print("Generating comprehensive report...")
    generate_report(feature_importances, pearson_correlations, spearman_correlations, execution_times, file_paths, df, output_dir)
    
    print("Analysis complete.")

if __name__ == "__main__":
    main()
