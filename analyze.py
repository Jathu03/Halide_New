import os
import pandas as pd
import numpy as np
import re
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
    """Find all files recursively in the given directory."""
    all_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # You might want to filter by file extension if needed
            # if filename.endswith('.json'):  # example filter
            all_files.append(os.path.join(dirpath, filename))
    return all_files

def extract_execution_time(file_content):
    """Extract execution time from file content."""
    # This function needs to be customized based on your file format
    # Example: Looking for patterns like "execution_time: 234ms" or "time: 1.2s"
    time_pattern = re.search(r'execution_time[:\s]+(\d+\.?\d*)[\s]?ms', file_content)
    if time_pattern:
        return float(time_pattern.group(1))
    
    # Try another pattern if the first one fails
    time_pattern = re.search(r'time[:\s]+(\d+\.?\d*)[\s]?ms', file_content)
    if time_pattern:
        return float(time_pattern.group(1))
    
    # Return None if no match is found
    return None

def extract_features(file_content):
    """Extract scheduling features from file content."""
    # This function needs to be customized based on your file format
    # Example: trying to parse JSON content
    try:
        data = json.loads(file_content)
        # Assuming features are in a top-level key or can be accessed somehow
        if 'features' in data:
            return data['features']
        else:
            # Return all data if specific features key isn't found
            # Filter out the execution time if it exists in the data
            if 'execution_time' in data:
                data_copy = data.copy()
                del data_copy['execution_time']
                return data_copy
            return data
    except json.JSONDecodeError:
        # If not JSON, try to extract features using regex
        features = {}
        # Example: look for patterns like "feature_name: value"
        feature_patterns = re.findall(r'(\w+):\s*([0-9.]+)', file_content)
        for name, value in feature_patterns:
            if name != 'execution_time' and name != 'time':  # Skip the time value
                try:
                    features[name] = float(value)
                except ValueError:
                    features[name] = value
        return features

def analyze_data(df):
    """Perform analysis on the collected data."""
    # Drop rows with missing execution times
    df = df.dropna(subset=['execution_time'])
    
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
    sns.barplot(x='RF_Importance', y='Feature', data=rf_feature_importance)
    plt.title('Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'rf_feature_importance.png'))
    
    # Plot correlation with execution time
    plt.figure(figsize=(12, 8))
    correlation_with_time_df = pd.DataFrame(correlation_with_time).drop('execution_time')
    sns.barplot(x=correlation_with_time_df['execution_time'], y=correlation_with_time_df.index)
    plt.title('Correlation with Execution Time')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'correlation_with_time.png'))
    
    # Plot execution time distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['execution_time'], kde=True)
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
        for _, row in analysis_results['rf_feature_importance'].iterrows():
            f.write(f"{row['Feature']}: {row['RF_Importance']:.4f}\n")
        f.write("\n")
        
        # Linear regression coefficients
        f.write("LINEAR REGRESSION COEFFICIENTS\n")
        f.write("============================\n")
        for _, row in analysis_results['lr_feature_importance'].iterrows():
            f.write(f"{row['Feature']}: Raw Coefficient = {row['LR_Coefficient']:.4f}, ")
            f.write(f"Standardized = {row['LR_Standardized_Coefficient']:.4f}\n")
        f.write("\n")
        
        # Correlation with execution time
        f.write("CORRELATION WITH EXECUTION TIME\n")
        f.write("==============================\n")
        for feature, corr in analysis_results['correlation_with_time'].items():
            if feature != 'execution_time':
                f.write(f"{feature}: {corr:.4f}\n")
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
        top_features = analysis_results['rf_feature_importance']['Feature'].tolist()[:3]
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
    
    # Initialize a list to store all the data
    data_rows = []
    
    # Process each file
    for file_path in all_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Extract execution time
            execution_time = extract_execution_time(content)
            
            # Extract features
            features = extract_features(content)
            
            # Skip files where we couldn't extract the required information
            if execution_time is None or not features:
                continue
            
            # Create a data row with features and execution time
            data_row = features.copy() if isinstance(features, dict) else {}
            data_row['execution_time'] = execution_time
            data_row['file_path'] = file_path
            
            data_rows.append(data_row)
            
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
    
    # Create a DataFrame from all the collected data
    print(f"Creating dataset from {len(data_rows)} processed files")
    if not data_rows:
        print("No valid data found. Please check the file format or extraction functions.")
        return
    
    df = pd.DataFrame(data_rows)
    
    # Perform analysis
    print("Analyzing data...")
    analysis_results = analyze_data(df)
    
    # Generate the report
    print("Generating report...")
    report_file = generate_report(analysis_results)
    
    print(f"Analysis complete! Report has been saved to '{report_file}'")
    print(f"Visualizations have been saved to the '{analysis_results['plots_dir']}' directory")

if __name__ == "__main__":
    main()
