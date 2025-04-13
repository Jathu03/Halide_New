import os
import json
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any, Union

class ExecutionDataProcessor:
    def __init__(self, main_dir: str, random_seed: int = 42, test_size: int = 50):
        """
        Initialize the data processor.
        
        Args:
            main_dir: Main directory containing subdirectories with JSON files
            random_seed: Random seed for reproducibility
            test_size: Number of samples to use for testing
        """
        self.main_dir = main_dir
        self.test_size = test_size
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Set random seed for reproducibility
        random.seed(random_seed)
    
    def get_execution_time(self, file_path: str) -> Optional[float]:
        """
        Extract execution time from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Execution time in milliseconds or None if not found
        """
        try:
            with open(file_path, 'rb') as f:
                raw_content = f.read()
                content = raw_content.decode('utf-8', errors='replace').replace('\0', '')
                data = json.loads(content)
            
            # First check in scheduling_data
            schedules = data.get("scheduling_data", [])
            for item in schedules:
                if isinstance(item, dict) and item.get('name') == 'total_execution_time_ms':
                    execution_time = item.get('value')
                    if execution_time is not None:
                        return float(execution_time)
            
            # Then check in programming_details.Schedules
            if 'programming_details' in data and 'Schedules' in data['programming_details']:
                for item in data['programming_details']['Schedules']:
                    if isinstance(item, dict) and item.get('Name') == 'total_execution_time_ms':
                        execution_time = item.get('Value', item.get('value'))
                        if execution_time is not None:
                            return float(execution_time)
            
            # Last resort: try to find any execution time in the last schedule
            if schedules and isinstance(schedules[-1], dict) and "value" in schedules[-1]:
                return float(schedules[-1]["value"])
            
            return None
        
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None
    
    def extract_features_from_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract features from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary of features or None if extraction failed
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            execution_time = self.get_execution_time(file_path)
            
            if execution_time is None:
                print(f"Warning: No execution time found in {file_path}")
                return None
            
            # Initialize feature containers
            nodes_features = []
            edges_features = []
            scheduling_features = []
            
            # Process programming details
            programming_details = data.get("programming_details", {})
            
            # Extract node features
            if 'Nodes' in programming_details:
                for node in programming_details['Nodes']:
                    node_feature = {'Name': node.get('Name', '')}
                    
                    # Extract operation histogram
                    if 'Details' in node and 'Op histogram' in node['Details']:
                        op_hist = node['Details']['Op histogram']
                        for op_line in op_hist:
                            parts = op_line.strip().split(':')
                            if len(parts) == 2:
                                op_name = parts[0].strip()
                                op_count = int(parts[1].strip())
                                node_feature[f'op_{op_name.lower()}'] = op_count
                    
                    # Extract node type information (improved feature)
                    if 'Type' in node:
                        node_feature['node_type'] = node['Type']
                    
                    # Extract node cost if available (improved feature)
                    if 'Details' in node and 'Cost' in node['Details']:
                        node_feature['node_cost'] = node['Details']['Cost']
                    
                    nodes_features.append(node_feature)
            
            # Extract edge features
            if 'Edges' in programming_details:
                for edge in programming_details['Edges']:
                    edge_feature = {
                        'From': edge.get('From', ''),
                        'To': edge.get('To', ''),
                        'Name': edge.get('Name', '')
                    }
                    
                    # Extract edge type information (improved feature)
                    if 'Type' in edge:
                        edge_feature['edge_type'] = edge['Type']
                    
                    # Extract data flow volume if available (improved feature)
                    if 'DataSize' in edge:
                        edge_feature['data_size'] = edge['DataSize']
                    
                    edges_features.append(edge_feature)
            
            # Extract scheduling features
            scheduling_data = data.get("scheduling_data", [])
            if not scheduling_data and 'Schedules' in programming_details:
                scheduling_data = programming_details['Schedules']
            
            for sched in scheduling_data:
                if not isinstance(sched, dict):
                    continue
                
                sched_feature = {'Name': sched.get('Name', sched.get('name', ''))}
                
                # Extract scheduling features directly 
                for key in ['bytes_at_production', 'bytes_at_realization', 'bytes_at_root', 
                           'bytes_at_task', 'inner_parallelism', 'outer_parallelism', 
                           'num_productions', 'num_realizations', 'num_scalars', 
                           'num_vectors', 'points_computed_total', 'working_set']:
                    if key in sched:
                        sched_feature[key] = sched[key]
                
                # Look in Details.scheduling_feature if present
                if 'Details' in sched and 'scheduling_feature' in sched['Details']:
                    sf = sched['Details']['scheduling_feature']
                    for key, value in sf.items():
                        sched_feature[key] = value
                
                scheduling_features.append(sched_feature)
            
            # Base features
            features = {
                'execution_time': execution_time,
                'nodes_count': len(nodes_features),
                'edges_count': len(edges_features),
                'scheduling_count': len(scheduling_features)
            }
            
            # Node-edge ratio
            if len(nodes_features) > 0 and len(edges_features) > 0:
                features['node_edge_ratio'] = len(nodes_features) / len(edges_features)
            else:
                features['node_edge_ratio'] = 0
            
            # Operation counts
            op_counts = {}
            for node in nodes_features:
                for key, value in node.items():
                    if key.startswith('op_'):
                        op_counts[key] = op_counts.get(key, 0) + value
            features.update(op_counts)
            
            # Node type distribution (improved feature)
            node_types = {}
            for node in nodes_features:
                if 'node_type' in node:
                    node_type = node['node_type']
                    node_types[f'nodetype_{node_type}'] = node_types.get(f'nodetype_{node_type}', 0) + 1
            features.update(node_types)
            
            # Edge type distribution (improved feature)
            edge_types = {}
            for edge in edges_features:
                if 'edge_type' in edge:
                    edge_type = edge['edge_type']
                    edge_types[f'edgetype_{edge_type}'] = edge_types.get(f'edgetype_{edge_type}', 0) + 1
            features.update(edge_types)
            
            # Aggregate scheduling metrics
            if scheduling_features:
                # Important scheduling metrics
                important_metrics = [
                    'bytes_at_production', 'bytes_at_realization', 'bytes_at_root', 'bytes_at_task',
                    'inner_parallelism', 'outer_parallelism', 'num_productions', 'num_realizations',
                    'num_scalars', 'num_vectors', 'points_computed_total', 'working_set'
                ]
                
                if scheduling_features and scheduling_features[0]:
                    for metric in important_metrics:
                        if metric in scheduling_features[0]:
                            features[f'sched_{metric}'] = scheduling_features[0][metric]
                
                # Calculate aggregate scheduling metrics
                total_bytes_at_production = sum(sf.get('bytes_at_production', 0) for sf in scheduling_features if isinstance(sf, dict))
                total_vectors = sum(sf.get('num_vectors', 0) for sf in scheduling_features if isinstance(sf, dict))
                total_parallelism = sum(sf.get('inner_parallelism', 0) * sf.get('outer_parallelism', 1) for sf in scheduling_features if isinstance(sf, dict))
                total_points = sum(sf.get('points_computed_total', 0) for sf in scheduling_features if isinstance(sf, dict))
                
                features['total_bytes_at_production'] = total_bytes_at_production
                features['total_vectors'] = total_vectors
                features['total_parallelism'] = total_parallelism
                features['total_points'] = total_points
                
                # Calculate derived metrics
                if total_vectors > 0:
                    features['bytes_per_vector'] = total_bytes_at_production / total_vectors
                else:
                    features['bytes_per_vector'] = 0
                
                if total_points > 0:
                    features['bytes_per_point'] = total_bytes_at_production / total_points
                else:
                    features['bytes_per_point'] = 0
                
                # Memory pressure
                if 'working_set' in scheduling_features[0] and 'bytes_at_production' in scheduling_features[0]:
                    if scheduling_features[0]['bytes_at_production'] > 0:
                        features['memory_pressure'] = scheduling_features[0]['working_set'] / scheduling_features[0]['bytes_at_production']
                    else:
                        features['memory_pressure'] = 0
                
                # Maximum memory usage (improved feature)
                features['max_memory'] = max((sf.get('working_set', 0) for sf in scheduling_features if isinstance(sf, dict)), default=0)
            
            # Calculate operation diversity metrics
            if len(nodes_features) > 0:
                op_types = sum(1 for k in op_counts.keys())
                features['avg_ops_per_node'] = sum(op_counts.values()) / len(nodes_features)
                features['op_diversity'] = op_types / len(nodes_features) if len(nodes_features) > 0 else 0
                
                # Add operation density (improved feature)
                if len(edges_features) > 0:
                    features['op_density'] = sum(op_counts.values()) / len(edges_features)
                else:
                    features['op_density'] = 0
            
            # Graph structure metrics (improved feature)
            if len(nodes_features) > 0 and len(edges_features) > 0:
                # Average degree
                features['avg_degree'] = 2 * len(edges_features) / len(nodes_features)
                
                # Estimate graph depth by analyzing from-to relationships
                node_connections = {}
                for edge in edges_features:
                    from_node = edge.get('From', '')
                    to_node = edge.get('To', '')
                    if from_node and to_node:
                        if from_node not in node_connections:
                            node_connections[from_node] = set()
                        node_connections[from_node].add(to_node)
                
                # Estimate graph depth through simple BFS
                if node_connections:
                    start_nodes = set(node_connections.keys()) - set([to for froms in node_connections.values() for to in froms])
                    if not start_nodes and node_connections:
                        start_nodes = {list(node_connections.keys())[0]}
                    
                    if start_nodes:
                        max_depth = 0
                        for start in start_nodes:
                            visited = {start}
                            frontier = [start]
                            depth = 0
                            
                            while frontier:
                                next_frontier = []
                                for node in frontier:
                                    for child in node_connections.get(node, set()):
                                        if child not in visited:
                                            visited.add(child)
                                            next_frontier.append(child)
                                
                                if next_frontier:
                                    depth += 1
                                    frontier = next_frontier
                                else:
                                    break
                            
                            max_depth = max(max_depth, depth)
                        
                        features['graph_depth'] = max_depth
            
            return features
        
        except Exception as e:
            print(f"Error extracting features from {file_path}: {str(e)}")
            return None
    
    def process_directory(self, directory_path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Process all JSON files in a directory.
        
        Args:
            directory_path: Path to the directory with JSON files
            
        Returns:
            Tuple of (features_list, file_names)
        """
        all_features = []
        file_names = []
        
        json_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.json')])
        
        for filename in json_files:
            file_path = os.path.join(directory_path, filename)
            features = self.extract_features_from_file(file_path)
            if features is not None:
                all_features.append(features)
                file_names.append(filename)
        
        return all_features, file_names
    
    def process_main_directory(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        """
        Process the main directory containing subdirectories with JSON files.
        
        Returns:
            Tuple of (train_features, test_features, test_file_names)
        """
        all_features = []
        all_file_names = []
        
        subdirs = sorted([d for d in os.listdir(self.main_dir) if os.path.isdir(os.path.join(self.main_dir, d))])
        
        if len(subdirs) < 1:
            raise ValueError(f"Expected at least 1 subdirectory in {self.main_dir}, found {len(subdirs)}")
        
        for subdir in subdirs:
            subdir_path = os.path.join(self.main_dir, subdir)
            features, file_names = self.process_directory(subdir_path)
            
            if not features:
                print(f"Skipping {subdir} due to no valid data")
                continue
            
            all_features.extend(features)
            all_file_names.extend([os.path.join(subdir, fname) for fname in file_names])
            print(f"Processed subdir {subdir}: {len(features)} files")
        
        total_files = len(all_features)
        if total_files < 50:
            raise ValueError(f"Expected at least 50 files total, found {total_files}")
        
        # Shuffle and split data
        combined = list(zip(all_features, all_file_names))
        random.shuffle(combined)
        all_features, all_file_names = zip(*combined)
        
        test_size = min(self.test_size, len(all_features) // 5)  # Ensure test set is not too large
        train_features = all_features[:-test_size]
        test_features = all_features[-test_size:]
        test_file_names = all_file_names[-test_size:]
        
        print(f"Total files: {total_files}")
        print(f"Training files: {len(train_features)}")
        print(f"Testing files: {len(test_features)}")
        
        return list(train_features), list(test_features), list(test_file_names)
    
    def clean_and_transform_features(self, train_features: List[Dict[str, Any]], 
                                    test_features: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Clean and transform features for model training.
        
        Args:
            train_features: List of training feature dictionaries
            test_features: List of test feature dictionaries
            
        Returns:
            Tuple of (train_df, test_df)
        """
        all_features_df = pd.DataFrame(train_features + test_features)
        
        # Fill NaN values
        all_features_df = all_features_df.fillna(0)
        
        # Remove constant columns
        constant_columns = [col for col in all_features_df.columns 
                           if col != 'execution_time' and all_features_df[col].nunique() == 1]
        all_features_df = all_features_df.drop(columns=constant_columns)
        print(f"Dropped {len(constant_columns)} constant columns")
        
        # Log transform execution time for stability
        if 'execution_time' in all_features_df.columns:
            all_features_df['execution_time_log'] = np.log1p(all_features_df['execution_time'])
        
        # Calculate additional feature: bytes per vector
        if 'total_vectors' in all_features_df.columns and all_features_df['total_vectors'].max() > 0:
            all_features_df['bytes_per_vector'] = all_features_df['total_bytes_at_production'] / (all_features_df['total_vectors'] + 1e-8)
        
        # Keep only numeric columns
        numeric_cols = all_features_df.select_dtypes(include=['number']).columns
        all_features_df = all_features_df[numeric_cols]
        
        # Split back into train and test
        train_size = len(train_features)
        train_df = all_features_df.iloc[:train_size]
        test_df = all_features_df.iloc[train_size:]
        
        return train_df, test_df
    
    def prepare_data_for_model(self, train_features: List[Dict[str, Any]], 
                              test_features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare data for model training, including scaling.
        
        Args:
            train_features: List of training feature dictionaries
            test_features: List of test feature dictionaries
        
        Returns:
            Dictionary containing processed data and metadata
        """
        train_df, test_df = self.clean_and_transform_features(train_features, test_features)
        
        # Determine if log transformed
        if 'execution_time_log' in train_df.columns:
            y_train = train_df['execution_time_log'].values.reshape(-1, 1)
            y_test = test_df['execution_time_log'].values.reshape(-1, 1)
            train_df = train_df.drop(['execution_time', 'execution_time_log'], axis=1)
            test_df = test_df.drop(['execution_time', 'execution_time_log'], axis=1)
            is_log_transformed = True
        else:
            y_train = train_df['execution_time'].values.reshape(-1, 1)
            y_test = test_df['execution_time'].values.reshape(-1, 1)
            train_df = train_df.drop('execution_time', axis=1)
            test_df = test_df.drop('execution_time', axis=1)
            is_log_transformed = False
        
        # Scale the data
        X_train_scaled = self.scaler_X.fit_transform(train_df)
        X_test_scaled = self.scaler_X.transform(test_df)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        # Print feature count
        feature_dimension = X_train_scaled.shape[1]
        print(f"Input feature dimension: {feature_dimension}")
        
        # Return as dictionary
        return {
            'X_train': X_train_scaled,
            'y_train': y_train_scaled,
            'X_test': X_test_scaled,
            'y_test': y_test_scaled,
            'y_train_raw': y_train,
            'y_test_raw': y_test,
            'scaler_y': self.scaler_y,
            'scaler_X': self.scaler_X,
            'feature_dimension': feature_dimension,
            'is_log_transformed': is_log_transformed,
            'train_df': train_df,
            'test_df': test_df
        }
    
    def create_dataset(self, save_to_file: bool = True) -> Dict[str, Any]:
        """
        Create the full dataset and optionally save it to disk.
        
        Args:
            save_to_file: Whether to save the processed data to files
            
        Returns:
            Dictionary containing all data and metadata
        """
        # Process the data
        train_features, test_features, test_file_names = self.process_main_directory()
        
        # Extract original execution times for comparison
        original_execution_times = {}
        for feature, fname in zip(test_features, test_file_names):
            original_execution_times[fname] = feature['execution_time']
        
        # Prepare data for model
        data_dict = self.prepare_data_for_model(train_features, test_features)
        
        # Add test file names and original execution times
        data_dict['test_file_names'] = test_file_names
        data_dict['original_execution_times'] = original_execution_times
        
        # Save to files if requested
        if save_to_file:
            print("Saving processed data to files...")
            
            # Create output directory if it doesn't exist
            os.makedirs('processed_data', exist_ok=True)
            
            # Save data
            np.save('processed_data/X_train.npy', data_dict['X_train'])
            np.save('processed_data/y_train.npy', data_dict['y_train'])
            np.save('processed_data/X_test.npy', data_dict['X_test'])
            np.save('processed_data/y_test.npy', data_dict['y_test'])
            
            # Save metadata
            metadata = {
                'feature_dimension': data_dict['feature_dimension'],
                'is_log_transformed': data_dict['is_log_transformed'],
                'test_file_names': test_file_names,
                'original_execution_times': original_execution_times,
            }
            
            with open('processed_data/metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save DataFrames to CSV
            data_dict['train_df'].to_csv('processed_data/train_features.csv', index=False)
            data_dict['test_df'].to_csv('processed_data/test_features.csv', index=False)
            
            print("Data successfully saved to 'processed_data/' directory")
        
        return data_dict

if __name__ == "__main__":
    processor = ExecutionDataProcessor(main_dir="synthetic_data", random_seed=42, test_size=50)
    dataset = processor.create_dataset(save_to_file=True)
    print(f"Dataset created with {dataset['feature_dimension']} features")
    print(f"Training samples: {dataset['X_train'].shape[0]}")
    print(f"Testing samples: {dataset['X_test'].shape[0]}")
    
    # Print feature importances using a quick Random Forest analysis
    try:
        from sklearn.ensemble import RandomForestRegressor
        import matplotlib.pyplot as plt
        
        print("\nAnalyzing feature importance...")
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(dataset['X_train'], dataset['y_train'].ravel())
        
        # Get feature importances
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Print feature ranking
        print("Feature ranking:")
        feature_names = dataset['train_df'].columns
        for i in range(min(20, dataset['feature_dimension'])):
            print(f"{i+1}. Feature {indices[i]} ({feature_names[indices[i]]}) - {importances[indices[i]]:.4f}")
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances")
        plt.bar(range(min(20, dataset['feature_dimension'])), 
                importances[indices[:20]], align="center")
        plt.xticks(range(min(20, dataset['feature_dimension'])), 
                  [feature_names[i] for i in indices[:20]], rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importances.png')
        plt.close()
        print("Feature importance plot saved as 'feature_importances.png'")
    except ImportError:
        print("scikit-learn or matplotlib not available for feature importance analysis")
