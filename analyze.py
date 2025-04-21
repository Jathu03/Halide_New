import os
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def find_all_files(root_dir):
    """Find all JSON files recursively in the given directory."""
    all_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.json'):  # Filter for JSON files
                all_files.append(os.path.join(dirpath, filename))
    return all_files

def extract_data_from_file(file_path):
    """Extract features and execution time from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Create a dictionary to store the data
        result = {'file_path': file_path}
        
        # Extract execution time from the specific structure you mentioned
        execution_time = None
        
        # Check if scheduling_data exists and contains a list of items
        if 'scheduling_data' in data and isinstance(data['scheduling_data'], list):
            for item in data['scheduling_data']:
                # Look for the execution time item with name "total_execution_time_ms"
                if isinstance(item, dict) and 'name' in item and 'value' in item:
                    if item['name'] == 'total_execution_time_ms':
                        execution_time = item['value']
                        break
                    # Also look for other possible execution time names
                    elif item['name'] in ['execution_time', 'runtime', 'time_ms', 'elapsed_time']:
                        execution_time = item['value']
                        # Keep searching in case we find the preferred "total_execution_time_ms" later
        
        # If we couldn't find execution time, print debug info and return None
        if execution_time is None:
            print(f"Warning: Could not find execution time in {file_path}")
            # Print more detailed structure to debug
            if 'scheduling_data' in data:
                print(f"Scheduling data contains {len(data['scheduling_data'])} items")
                for i, item in enumerate(data['scheduling_data'][:5]):  # Print first 5 for debugging
                    if isinstance(item, dict) and 'name' in item:
                        print(f"  Item {i}: name={item['name']}")
            return None
        
        result['execution_time'] = float(execution_time)
        
        # Extract features from programming_details if available
        if 'programming_details' in data and isinstance(data['programming_details'], dict):
            for key, value in data['programming_details'].items():
                if isinstance(value, (int, float, bool)):
                    result[f"prog_{key}"] = value
                elif isinstance(value, str) and value.isdigit():
                    result[f"prog_{key}"] = float(value)
        
        # Extract additional features from scheduling_data
        if 'scheduling_data' in data and isinstance(data['scheduling_data'], list):
            for item in data['scheduling_data']:
                if isinstance(item, dict) and 'name' in item and 'value' in item:
                    # Skip the execution time we already extracted
                    if item['name'] == 'total_execution_time_ms':
                        continue
                    
                    # Try to convert value to numeric if possible
                    try:
                        if isinstance(item['value'], (int, float)):
                            result[f"sched_{item['name']}"] = float(item['value'])
                        elif isinstance(item['value'], str) and item['value'].replace('.', '', 1).isdigit():
                            result[f"sched_{item['name']}"] = float(item['value'])
                    except:
                        # If conversion fails, skip this item
                        pass
        
        return result
    
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def analyze_data(df):
    """Perform analysis on the collected data."""
    # Drop rows with missing execution times
    df = df.dropna(subset=['execution_time'])
    
    # Print basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Execution time range: {df['execution_time'].min()} - {df['execution_time'].max()} ms")
    
    # Convert all feature columns to numeric where possible
    for col in df.columns:
        if col != 'execution_time' and col != 'file_path':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    
    # Drop non-numeric columns (except file_path which we'll keep for reference)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != 'execution_time']
    
    if not feature_cols:
        return "No numeric features found for analysis."
    
    print(f"Numeric features for analysis: {len(feature_cols)}")
    print(f"Sample features: {feature_cols[:10]}")
    
    # Split data
    X = df[feature_cols]
    y = df['execution_time']
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest model for feature importance
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Feature importance from Random Forest
    rf_feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'RF_Importance': rf_model.feature_importances_
    }).sort_values('RF_Importance', ascending=False)
    
    # Linear regression for coefficients
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Calculate standardized coefficients
    lr_importance = np.abs(lr_model.coef_ * np.std(X, axis=0))
    
    # Linear regression coefficients
    lr_feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'LR_Coefficient': lr_model.coef_,
        'LR_Standardized_Coefficient': lr_importance
    }).sort_values('LR_Standardized_Coefficient', ascending=False)
    
    # Evaluate models
    rf_predictions = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_predictions)
    rf_r2 = r2_score(y_test, rf_predictions)
    
    lr_predictions = lr_model.predict(X_test)
    lr_mse = mean_squared_error(y_test, lr_predictions)
    lr_r2 = r2_score(y_test, lr_predictions)
    
    # Create correlation matrix
    correlation_matrix = df[numeric_cols].corr()
    correlation_with_time = correlation_matrix['execution_time'].sort_values(ascending=False)
    
    # Basic statistics
    basic_stats = df.describe()
    
    # Create plots directory if it doesn't exist
    plots_dir = "feature_analysis_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_n = min(20, len(rf_feature_importance))  # Limit to top 20 features
    sns.barplot(x='RF_Importance', y='Feature', data=rf_feature_importance.head(top_n))
    plt.title('Top Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'rf_feature_importance.png'))
    
    # Plot correlation with execution time
    plt.figure(figsize=(12, 8))
    correlation_with_time_df = pd.DataFrame(correlation_with_time).drop('execution_time')
    top_corr = correlation_with_time_df.head(top_n)
    sns.barplot(x=top_corr['execution_time'], y=top_corr.index)
    plt.title('Top Correlations with Execution Time')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'correlation_with_time.png'))
    
    # Plot execution time distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['execution_time'], kde=True, bins=50)
    plt.title('Execution Time Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'execution_time_distribution.png'))
    
    # Scatter plots for top features
    for feature in rf_feature_importance['Feature'][:5]:  # Top 5 features
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[feature], y=df['execution_time'])
        plt.title(f'Execution Time vs {feature}')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'scatter_{feature}.png'))
    
    return {
        'rf_feature_importance': rf_feature_importance,
        'lr_feature_importance': lr_feature_importance,
        'correlation_with_time': correlation_with_time,
        'basic_stats': basic_stats,
        'rf_mse': rf_mse,
        'rf_r2': rf_r2,
        'lr_mse': lr_mse,
        'lr_r2': lr_r2,
        'plots_dir': plots_dir
    }

def generate_report(analysis_results, output_file="execution_time_analysis_report.txt"):
    """Generate a comprehensive report from the analysis results."""
    with open(output_file, 'w') as f:
        f.write("============================================\n")
        f.write("EXECUTION TIME FEATURE ANALYSIS REPORT\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("============================================\n\n")
        
        f.write("SUMMARY\n")
        f.write("=======\n")
        f.write("This report analyzes the relationship between various scheduling features\n")
        f.write("and execution times across multiple files in the dataset.\n\n")
        
        # Model performance
        f.write("MODEL PERFORMANCE\n")
        f.write("================\n")
        f.write(f"Random Forest Mean Squared Error: {analysis_results['rf_mse']:.4f}\n")
        f.write(f"Random Forest R² Score: {analysis_results['rf_r2']:.4f}\n")
        f.write(f"Linear Regression Mean Squared Error: {analysis_results['lr_mse']:.4f}\n")
        f.write(f"Linear Regression R² Score: {analysis_results['lr_r2']:.4f}\n\n")
        
        # Feature importance from Random Forest
        f.write("FEATURE IMPORTANCE (RANDOM FOREST)\n")
        f.write("================================\n")
        for _, row in analysis_results['rf_feature_importance'].head(20).iterrows():
            f.write(f"{row['Feature']}: {row['RF_Importance']:.4f}\n")
        f.write("\n")
        
        # Linear regression coefficients
        f.write("LINEAR REGRESSION COEFFICIENTS (TOP 20)\n")
        f.write("====================================\n")
        for _, row in analysis_results['lr_feature_importance'].head(20).iterrows():
            f.write(f"{row['Feature']}: Raw Coefficient = {row['LR_Coefficient']:.4f}, ")
            f.write(f"Standardized = {row['LR_Standardized_Coefficient']:.4f}\n")
        f.write("\n")
        
        # Correlation with execution time
        f.write("CORRELATION WITH EXECUTION TIME (TOP 20)\n")
        f.write("====================================\n")
        count = 0
        for feature, corr in analysis_results['correlation_with_time'].items():
            if feature != 'execution_time':
                f.write(f"{feature}: {corr:.4f}\n")
                count += 1
                if count >= 20:
                    break
        f.write("\n")
        
        # Basic statistics
        f.write("BASIC STATISTICS\n")
        f.write("===============\n")
        f.write(str(analysis_results['basic_stats']) + "\n\n")
        
        # Visualization information
        f.write("VISUALIZATIONS\n")
        f.write("=============\n")
        f.write(f"Plots have been saved to the '{analysis_results['plots_dir']}' directory.\n")
        f.write("Available visualizations:\n")
        f.write("- Random Forest Feature Importance\n")
        f.write("- Correlation with Execution Time\n")
        f.write("- Execution Time Distribution\n")
        f.write("- Scatter plots for top features\n\n")
        
        f.write("CONCLUSION\n")
        f.write("==========\n")
        
        # Get top features for conclusion
        top_features = analysis_results['rf_feature_importance']['Feature'].tolist()[:5]
        f.write(f"The analysis indicates that the most significant features affecting execution time are:\n")
        for i, feature in enumerate(top_features, 1):
            importance = analysis_results['rf_feature_importance'].loc[
                analysis_results['rf_feature_importance']['Feature'] == feature, 'RF_Importance'
            ].values[0]
            f.write(f"{i}. {feature} (Importance: {importance:.4f})\n")
        
        f.write("\nThese features should be prioritized when optimizing scheduling algorithms for performance.\n")
    
    return output_file

def main():
    # Path to the main directory
    root_dir = "synthetic_data"
    
    # Find all files
    print(f"Scanning for files in {root_dir}...")
    all_files = find_all_files(root_dir)
    print(f"Found {len(all_files)} files")
    
    # Process each file and collect data
    data_rows = []
    for i, file_path in enumerate(all_files):
        result = extract_data_from_file(file_path)
        if result:
            data_rows.append(result)
        
        # Print progress
        if (i+1) % 500 == 0 or i+1 == len(all_files):
            print(f"Processed {i+1}/{len(all_files)} files, collected {len(data_rows)} valid entries")
    
    print(f"Creating dataset from {len(data_rows)} processed files")
    if not data_rows:
        print("No valid data found. Please check the file format or extraction functions.")
        return
    
    # Create DataFrame from collected data
    df = pd.DataFrame(data_rows)
    
    # Save the raw data for later reference
    df.to_csv("raw_execution_time_data.csv", index=False)
    
    # Perform analysis
    print("Analyzing data...")
    analysis_results = analyze_data(df)
    
    if isinstance(analysis_results, str):
        print(f"Analysis failed: {analysis_results}")
        return
    
    # Generate the report
    print("Generating report...")
    report_file = generate_report(analysis_results)
    
    print(f"Analysis complete! Report has been saved to '{report_file}'")
    print(f"Visualizations have been saved to the '{analysis_results['plots_dir']}' directory")
    print(f"Raw data has been saved to 'raw_execution_time_data.csv'")

if __name__ == "__main__":
    main()
