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
import random
warnings.filterwarnings('ignore')

def extract_features_from_file(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        without_extern = data.get('without_extern', {})
        global_features = without_extern.get('global_features', {})
        
        execution_time = global_features.get('execution_time_ms', None)
        if execution_time is None or not np.isfinite(execution_time) or execution_time < 0:
            print(f"Invalid or missing execution time in {file_path}, using fallback value 0.001 ms")
            execution_time = 0.001
        elif execution_time == 0:
            print(f"Warning: {file_path} has execution_time_ms = 0, proceeding with value")
        
        nodes = without_extern.get('nodes', [])
        edges = without_extern.get('edges', [])
        
        features = {
            'execution_time': float(execution_time),
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
    invalid_files = []
    zero_time_files = []
    
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
            if features['execution_time'] == 0 or features['execution_time'] == 0.001:
                zero_time_files.append(str(file_path))
        else:
            invalid_files.append(str(file_path))
    
    print(f"Processed {len(all_features)} files successfully.           ")
    
    log_path = main_dir_path / 'invalid_files_log.txt'
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("Files skipped due to invalid execution times or errors:\n")
        for file_path in invalid_files:
            f.write(f"{file_path}\n")
        f.write("\nFiles with execution_time_ms = 0 or fallback value (included but potentially problematic):\n")
        for file_path in zero_time_files:
            f.write(f"{file_path}\n")
    
    print(f"Total files found: {len(all_features) + len(invalid_files)}")
    print(f"Files skipped: {len(invalid_files)}")
    print(f"Files with zero or fallback execution time: {len(zero_time_files)}")
    print(f"Valid files retained: {len(all_features)}")
    
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
    
    return feature_importances, pearson_correlations, spearman_correlations, y, df, X, rf, scaler, X.columns

def predict_execution_times(model, scaler, file_paths, main_dir, feature_names):
    predictions = []
    main_dir_path = Path(main_dir)
    
    print("\nPredicting execution times for selected files:")
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
        
        # Ensure X has the same columns as the training data
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
            'actual_time_ms': actual_time,
            'predicted_time_ms': predicted_time,
            'error_percentage': error_percentage
        })
        
        print(f"File: {file_path}")
        print(f"Actual Time: {actual_time:.2f} ms")
        print(f"Predicted Time: {predicted_time:.2f} ms")
        print(f"Error Percentage: {error_percentage:.2f}%")
        print()
    
    if predictions:
        mean_error_percentage = np.mean([p['error_percentage'] for p in predictions if p['error_percentage'] != float('inf')])
        print(f"Mean Absolute Percentage Error (MAPE): {mean_error_percentage:.2f}%")
    
    return predictions

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
    
    top_feature_names = list(feature_importances.index[:10])
    
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
    
    for i, (feature, importance) in enumerate(feature_importances.items(), start=1):
        if i > 20:
            break
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
    
    for i, (feature, corr) in enumerate(pearson_correlations.items(), start=1):
        if i > 20:
            break
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

def main(main_dir="Graph_Output", output_dir="analysis_results"):
    print(f"Processing files in {main_dir}...")
    features_list, file_paths = process_all_files(main_dir)
    
    if not features_list:
        print("No valid data found in the files. Check Graph_Output/invalid_files_log.txt for details.")
        return
    
    print(f"Extracted features from {len(features_list)} files.")
    
    print("Analyzing feature importance...")
    feature_importances, pearson_correlations, spearman_correlations, execution_times, df, X, rf, scaler, feature_names = analyze_feature_importance(features_list)
    
    print("Generating comprehensive report...")
    generate_report(feature_importances, pearson_correlations, spearman_correlations, execution_times, file_paths, df, output_dir)
    
    if len(file_paths) >= 5:
        selected_files = random.sample(file_paths, 5)
    else:
        selected_files = file_paths
        print(f"Only {len(file_paths)} files available, using all for prediction.")
    
    predictions = predict_execution_times(rf, scaler, selected_files, main_dir, feature_names)
    
    with open(f"{output_dir}/feature_importance_report.txt", 'a') as f:
        f.write("\nExecution Time Predictions:\n")
        f.write("--------------------------\n")
        for pred in predictions:
            f.write(f"File: {pred['file']}\n")
            f.write(f"Actual Time: {pred['actual_time_ms']:.2f} ms\n")
            f.write(f"Predicted Time: {pred['predicted_time_ms']:.2f} ms\n")
            f.write(f"Error Percentage: {pred['error_percentage']:.2f}%\n\n")
        if predictions:
            mean_mape = np.mean([p['error_percentage'] for p in predictions if p['error_percentage'] != float('inf')])
            f.write(f"Mean Absolute Percentage Error (MAPE): {mean_mape:.2f}%\n")
    
    print("Analysis and predictions complete.")

if __name__ == "__main__":
    main()
