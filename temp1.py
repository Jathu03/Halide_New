import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import random
from collections import defaultdict
import matplotlib.pyplot as plt

# Define fixed sets of features
PROGRAM_FEATURES = [
    'op_add', 'op_sub', 'op_mul', 'op_div', 'op_mod', 'op_eq', 'op_ne', 'op_lt', 'op_le',
    'op_or', 'op_and', 'op_not', 'op_min', 'op_max', 'op_constant', 'op_variable',
    'op_funccall', 'op_imagecall', 'op_externcall', 'op_let', 'op_param',
    'memory_transpose_0', 'memory_transpose_1', 'memory_transpose_2', 'memory_transpose_3',
    'memory_slice_0', 'memory_slice_1', 'memory_slice_2', 'memory_slice_3',
    'memory_broadcast_0', 'memory_broadcast_1', 'memory_broadcast_2', 'memory_broadcast_3',
    'memory_pointwise_0', 'memory_pointwise_1', 'memory_pointwise_2', 'memory_pointwise_3',
    'op_diversity', 'nodes_count', 'edges_count', 'node_edge_ratio'
]

SCHEDULE_FEATURES = [
    'cache_hits', 'cache_misses', 'execution_time_ms', 'sched_num_realizations',
    'sched_num_productions', 'sched_points_computed_total', 'sched_innermost_loop_extent',
    'sched_inner_parallelism', 'sched_outer_parallelism', 'sched_bytes_at_realization',
    'sched_bytes_at_production', 'sched_bytes_at_root', 'sched_unique_bytes_read_per_realization',
    'sched_working_set', 'sched_vector_size', 'sched_num_vectors', 'sched_num_scalars',
    'sched_bytes_at_task', 'sched_working_set_at_task', 'sched_working_set_at_production',
    'sched_working_set_at_realization', 'sched_working_set_at_root', 'total_parallelism',
    'scheduling_count', 'total_bytes_at_production', 'total_vectors', 'computation_efficiency',
    'memory_pressure', 'memory_utilization_ratio', 'bytes_processing_rate', 'bytes_per_parallelism',
    'bytes_per_vector', 'nodes_per_schedule'
]

# Feature extraction function
def extract_features(json_data):
    program_features = {}
    schedule_features = {}
    
    # Extract global features (schedule-related)
    global_node = next((child for child in json_data['children'] if child['name'] == 'Global Features'), None)
    if global_node:
        schedule_features['cache_hits'] = global_node.get('cache_hits', 0)
        schedule_features['cache_misses'] = global_node.get('cache_misses', 0)
        schedule_features['execution_time_ms'] = global_node.get('execution_time_ms', 0)
    
    # Extract op_histogram features (program-related)
    op_histogram = defaultdict(int)
    for node in json_data['children']:
        if 'op_histogram' in node:
            for op, count in node['op_histogram'].items():
                op_histogram[op.lower()] += count
    for op, count in op_histogram.items():
        program_features[f'op_{op.lower()}'] = count
    
    # Extract memory patterns (program-related)
    memory_patterns = defaultdict(lambda: [0, 0, 0, 0])
    for node in json_data['children']:
        if 'memory_patterns' in node:
            for pattern, values in node['memory_patterns'].items():
                memory_patterns[pattern] = [sum(x) for x in zip(memory_patterns[pattern], values)]
    for pattern, values in memory_patterns.items():
        for i, val in enumerate(values):
            program_features[f'memory_{pattern.lower()}_{i}'] = val
    
    # Extract scheduling features (schedule-related)
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
        if key in ['inner_parallelism', 'outer_parallelism'] and node_count > 0:
            schedule_features[f'sched_{key}'] = scheduling_sums[key] / node_count
        else:
            schedule_features[f'sched_{key}'] = scheduling_sums[key]
    
    # Derived features (schedule-related)
    schedule_features['total_parallelism'] = schedule_features.get('sched_inner_parallelism', 0) + schedule_features.get('sched_outer_parallelism', 0)
    schedule_features['scheduling_count'] = schedule_features.get('sched_num_realizations', 0) + schedule_features.get('sched_num_productions', 0)
    schedule_features['total_bytes_at_production'] = schedule_features.get('sched_bytes_at_production', 0)
    schedule_features['total_vectors'] = schedule_features.get('sched_num_vectors', 0)
    schedule_features['computation_efficiency'] = (schedule_features.get('sched_points_computed_total', 0) /
                                                 schedule_features.get('sched_bytes_at_realization', 1)) if schedule_features.get('sched_bytes_at_realization', 0) != 0 else 0
    schedule_features['memory_pressure'] = (schedule_features.get('sched_working_set', 0) /
                                          schedule_features.get('sched_bytes_at_root', 1)) if schedule_features.get('sched_bytes_at_root', 0) != 0 else 0
    schedule_features['memory_utilization_ratio'] = (schedule_features.get('sched_unique_bytes_read_per_realization', 0) /
                                                   schedule_features.get('sched_bytes_at_task', 1)) if schedule_features.get('sched_bytes_at_task', 0) != 0 else 0
    schedule_features['bytes_processing_rate'] = (schedule_features.get('sched_bytes_at_realization', 0) /
                                                schedule_features.get('execution_time_ms', 1)) if schedule_features.get('execution_time_ms', 0) != 0 else 0
    schedule_features['bytes_per_parallelism'] = (schedule_features.get('sched_bytes_at_task', 0) /
                                                schedule_features.get('total_parallelism', 1)) if schedule_features.get('total_parallelism', 0) != 0 else 0
    schedule_features['bytes_per_vector'] = (schedule_features.get('sched_bytes_at_realization', 0) /
                                           schedule_features.get('sched_num_vectors', 1)) if schedule_features.get('sched_num_vectors', 0) != 0 else 0
    nodes_count = len(json_data['children'])
    edges_count = sum(len(node.get('children', [])) for node in json_data['children'])
    program_features['nodes_count'] = nodes_count
    program_features['edges_count'] = edges_count
    program_features['node_edge_ratio'] = nodes_count / (edges_count + 1)
    schedule_features['nodes_per_schedule'] = nodes_count / (schedule_features.get('scheduling_count', 1)) if schedule_features.get('scheduling_count', 0) != 0 else 0
    program_features['op_diversity'] = len([k for k, v in program_features.items() if k.startswith('op_') and v > 0])
    
    # Create fixed-length feature vectors
    fixed_program_features = {key: program_features.get(key, 0.0) for key in PROGRAM_FEATURES}
    fixed_schedule_features = {key: schedule_features.get(key, 0.0) for key in SCHEDULE_FEATURES}
    return fixed_program_features, fixed_schedule_features

# Process Tree_Output directory
def process_tree_output_directory(main_dir):
    all_program_features = []
    all_schedule_features = []
    file_names = []
    invalid_files = []
    
    for root, dirs, files in os.walk(main_dir):
        if 'tree_representation.json' in files:
            file_path = os.path.join(root, 'tree_representation.json')
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                program_features, schedule_features = extract_features(json_data)
                if schedule_features['execution_time_ms'] <= 0 or not np.isfinite(schedule_features['execution_time_ms']):
                    invalid_files.append(file_path)
                    print(f"Skipped file with invalid execution time: {file_path}")
                    continue
                all_program_features.append(program_features)
                all_schedule_features.append(schedule_features)
                file_names.append(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                invalid_files.append(file_path)
    
    if not all_program_features:
        raise ValueError("No valid JSON files with valid execution times found in Tree_Output directory.")
    
    log_path = os.path.join(main_dir, 'invalid_files_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("Files with invalid execution times or errors (skipped):\n")
        for file_path in invalid_files:
            f.write(f"{file_path}\n")
    
    total_files = len(all_program_features)
    print(f"Total valid files found: {total_files}")
    print(f"Files skipped due to invalid execution times or errors: {len(invalid_files)}")
    if total_files < 50:
        raise ValueError(f"Expected at least 50 valid files, found {total_files}")
    
    combined = list(zip(all_program_features, all_schedule_features, file_names))
    random.shuffle(combined)
    all_program_features, all_schedule_features, file_names = zip(*combined)
    
    test_size = min(50, total_files)
    train_program_features = all_program_features[:-test_size]
    train_schedule_features = all_schedule_features[:-test_size]
    test_program_features = all_program_features[-test_size:]
    test_schedule_features = all_schedule_features[-test_size:]
    train_file_names = file_names[:-test_size]
    test_file_names = file_names[-test_size:]
    
    print(f"Training files: {len(train_program_features)}")
    print(f"Testing files: {len(test_program_features)}")
    
    return train_program_features, train_schedule_features, test_program_features, test_schedule_features, list(test_file_names)

# Prepare data for model
def prepare_data_for_model(train_program_features, train_schedule_features, test_program_features, test_schedule_features):
    important_features = [
        'cache_hits', 'bytes_processing_rate', 'sched_bytes_at_task', 'sched_working_set_at_root',
        'sched_bytes_at_realization', 'sched_unique_bytes_read_per_realization'
    ]
    
    # Create DataFrames
    train_program_df = pd.DataFrame(train_program_features)
    train_schedule_df = pd.DataFrame(train_schedule_features)
    test_program_df = pd.DataFrame(test_program_features)
    test_schedule_df = pd.DataFrame(test_schedule_features)
    
    # Drop low-importance features
    low_importance_features = [
        'op_cast', 'op_selfcall', 'memory_pointwise_1', 'memory_transpose_1', 'memory_broadcast_1',
        'memory_slice_1', 'op_select', 'op_not', 'op_and', 'op_ne', 'op_mod', 'memory_pointwise_2',
        'memory_broadcast_2', 'memory_slice_2', 'memory_transpose_2', 'op_externcall', 'op_imagecall',
        'op_param', 'memory_pointwise_3', 'memory_transpose_3', 'op_sub', 'memory_pointwise_0', 'op_let'
    ]
    train_program_df = train_program_df.drop(columns=[col for col in low_importance_features if col in train_program_df.columns])
    test_program_df = test_program_df.drop(columns=[col for col in low_importance_features if col in test_program_df.columns])
    train_schedule_df = train_schedule_df.drop(columns=[col for col in low_importance_features if col in train_schedule_df.columns])
    test_schedule_df = test_schedule_df.drop(columns=[col for col in low_importance_features if col in test_schedule_df.columns])
    
    # Log transform skewed features
    skewed_features = ['cache_hits', 'bytes_processing_rate', 'sched_bytes_at_task', 'computation_efficiency']
    for feature in skewed_features:
        if feature in train_schedule_df.columns:
            train_schedule_df[f'log_{feature}'] = np.log1p(train_schedule_df[feature])
            test_schedule_df[f'log_{feature}'] = np.log1p(test_schedule_df[feature])
            train_schedule_df = train_schedule_df.drop(columns=[feature])
            test_schedule_df = test_schedule_df.drop(columns=[feature])
    
    train_program_df = train_program_df.fillna(0)
    test_program_df = test_program_df.fillna(0)
    train_schedule_df = train_schedule_df.fillna(0)
    test_schedule_df = test_schedule_df.fillna(0)
    
    # Remove constant columns
    constant_columns_program = [col for col in train_program_df.columns if train_program_df[col].nunique() == 1]
    train_program_df = train_program_df.drop(columns=constant_columns_program)
    test_program_df = test_program_df.drop(columns=constant_columns_program)
    
    constant_columns_schedule = [col for col in train_schedule_df.columns if train_schedule_df[col].nunique() == 1]
    train_schedule_df = train_schedule_df.drop(columns=constant_columns_schedule)
    test_schedule_df = test_schedule_df.drop(columns=constant_columns_schedule)
    
    # Extract execution times
    y_train_raw = np.array([f['execution_time_ms'] for f in train_schedule_features])
    y_test_raw = np.array([f['execution_time_ms'] for f in test_schedule_features])
    y_train_raw = np.clip(y_train_raw, 0, np.percentile(y_train_raw, 99))
    y_test_raw = np.clip(y_test_raw, 0, np.percentile(y_test_raw, 99))
    
    y_train = np.log1 # Modified main function to save the model
def main(main_dir):
    if torch.cuda.is_available():
        torch.cuda.init()
        print(f"CUDA initialized. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Using CPU.")
    
    print(f"Processing main directory: {main_dir}")
    train_program_features, train_schedule_features, test_program_features, test_schedule_features, test_file_names = process_tree_output_directory(main_dir)
    
    if len(train_program_features) == 0 or len(test_program_features) == 0:
        print("Error: No valid training or test data found")
        return None
    
    (train_program_tensor, train_schedule_tensor, y_train,
     test_program_tensor, test_schedule_tensor, y_test,
     y_scaler, program_input_size, schedule_input_size, program_columns, schedule_columns) = prepare_data_for_model(
        train_program_features, train_schedule_features, test_program_features, test_schedule_features
    )
    
    train_loader, test_loader = create_data_loaders(
        train_program_tensor, train_schedule_tensor, y_train,
        test_program_tensor, test_schedule_tensor, y_test,
        batch_size=64
    )
    
    global model
    model = SimpleLSTMModel(
        program_input_size=program_input_size,
        schedule_input_size=schedule_input_size,
        hidden_sizes=[512, 256, 128],
        output_size=1,
        dropout_rate=0.2,
        num_heads=8
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=1e-4)
    
    feature_importances = {
        'cache_hits': 0.5860,
        'bytes_processing_rate': 0.2893,
        'sched_bytes_at_task': 0.0422,
        'sched_working_set_at_root': 0.0248,
        'sched_bytes_at_realization': 0.0055,
        'sched_unique_bytes_read_per_realization': 0.0049
    }
    
    feature_indices = {}
    for feature in feature_importances.keys():
        log_feature = f'log_{feature}' if feature in ['cache_hits', 'bytes_processing_rate'] else feature
        if log_feature in schedule_columns:
            feature_indices[feature] = schedule_columns.get_loc(log_feature)
        else:
            feature_indices[feature] = schedule_columns.get_loc(feature) if feature in schedule_columns else -1
    
    print("Building and training Simple LSTM model...")
    train_losses, val_losses = train_model(
        model, train_loader, test_loader,
        custom_loss, optimizer, feature_indices, feature_importances,
        num_epochs=1000, patience=50, accumulation_steps=2
    )
    
    if train_losses is None or val_losses is None:
        print("Training failed due to invalid values")
        return None
    
    torch.save(model.state_dict(), "model.pt")
    print("Model saved to model.pt")
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(
        model, test_program_tensor, test_schedule_tensor, y_test,
        y_scaler, test_file_names
    )
    
    print(f"\nSummary for Comparison:")
    print(f"Model: SimpleLSTM")
    
    return model, y_scaler, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "Tree_Output"
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    model, y_scaler, y_test_actual, y_pred_actual = main(main_dir)
