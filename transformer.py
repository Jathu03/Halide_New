import json
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# Expanded vocabulary for categorical features
vocab = {
    'op_types': ['Constant', 'Cast', 'Variable', 'Param', 'Add', 'Sub', 'Mul', 'Div', 'Min', 'Max', 'EQ', 'NE', 'LT', 'LE', 'And', 'Or', 'Not', 'Select', 'ImageCall', 'FuncCall', 'SelfCall', 'ExternCall', 'Let'],
    'iterator_names': ['i0', 'i1', 'i2', 'i3'],
    'transformation_types': ['shiftings', 'tiling', 'unrolling_factor', 'parallelized_dim'],
    'access_types': ['Pointwise', 'Strided', 'Indirect'],
    'comp_names': ['comp00', 'comp01', 'comp02'],
    'node_names': []  # Dynamically populated
}
vocab_sizes = {k: len(v) for k, v in vocab.items()}

# Helper functions
def embed_categorical(value, vocab_key, max_size=10):
    if value not in vocab[vocab_key] and vocab_key == 'node_names' and len(vocab[vocab_key]) < max_size:
        vocab[vocab_key].append(value)
        vocab_sizes[vocab_key] = len(vocab[vocab_key])
    idx = vocab[vocab_key].index(value) if value in vocab[vocab_key] else -1
    one_hot = np.zeros(min(max_size, vocab_sizes[vocab_key]))
    if idx >= 0:
        one_hot[idx] = 1
    return one_hot.tolist()  # Return as list for concatenation

def normalize_numerical(values, target_length):
    values = values + [0.0] * (target_length - len(values)) if len(values) < target_length else values[:target_length]
    scaler = MinMaxScaler()
    values = np.array(values).reshape(-1, 1)
    normalized = scaler.fit_transform(values).flatten()
    return normalized.tolist()  # Return as list

def parse_number(x):
    try:
        if '/' in x:
            return float(eval(x))
        return float(x)
    except (ValueError, SyntaxError, NameError):
        return 0.0

# Process a single Halide JSON file
def process_halide(halide_data):
    edges = halide_data.get('programming_details', {}).get('Edges', [])
    nodes = halide_data.get('programming_details', {}).get('Nodes', [])
    sched_data = halide_data.get('scheduling_data', [])

    sequence = []
    type_dim = 3  # Node, Edge, Schedule
    specific_dim = 40  # name(10) + access(3) + ops(10) + jacobian(10) + sched(7)
    feature_dim = type_dim + specific_dim  # 3 + 40 = 43

    # Process nodes
    for node in nodes:
        type_onehot = [1.0, 0.0, 0.0]
        specific_features = []
        name = node.get('Name', 'unknown')
        details = node.get('Details', {})
        specific_features.extend(embed_categorical(name, 'node_names'))  # 10
        access_patterns = details.get('Memory access patterns', ['Pointwise: 1'])
        access = ' '.join(access_patterns[:2]).split(':')[1].strip().split()[0]
        specific_features.extend(embed_categorical(access, 'access_types'))  # 3
        op_hist = details.get('Op histogram', ['Add: 0'] * 10)
        op_counts = [int(line.split(':')[1].strip()) for line in op_hist[:10]]
        specific_features.extend(normalize_numerical(op_counts, 10))  # 10
        specific_features.extend([0.0] * 10)  # Jacobian placeholder
        sched = next((s.get('Details', {}).get('scheduling_feature', {}) for s in sched_data if s.get('Name') == name), {})
        sched_vals = [
            sched.get('inner_parallelism', 0),
            sched.get('outer_parallelism', 0),
            sched.get('vector_size', 0),
            sched.get('unrolled_loop_extent', 0),
            sched.get('points_computed_minimum', 0),
            sched.get('unique_bytes_read_per_realization', 0),
            sched.get('bytes_at_task', 0)
        ]
        specific_features.extend(normalize_numerical(sched_vals, 7))  # 7
        # Ensure exact length
        specific_features = specific_features[:specific_dim] + [0.0] * (specific_dim - len(specific_features))
        feature_vector = type_onehot + specific_features
        sequence.append(np.array(feature_vector))

    # Process edges
    for edge in edges:
        type_onehot = [0.0, 1.0, 0.0]
        specific_features = []
        from_to = f"{edge.get('From', 'unknown')}_{edge.get('To', 'unknown')}"
        specific_features.extend(embed_categorical(from_to, 'node_names'))  # 10
        specific_features.extend([0.0] * 3)  # Access placeholder
        specific_features.extend([0.0] * 10)  # Op histogram placeholder
        jacobian_str = ' '.join(edge.get('Details', {}).get('Load Jacobians', ['0'] * 10))
        jacobian = [parse_number(x) for x in jacobian_str.split()]
        specific_features.extend(normalize_numerical(jacobian, 10))  # 10
        specific_features.extend([0.0] * 7)  # Schedule placeholder
        # Ensure exact length
        specific_features = specific_features[:specific_dim] + [0.0] * (specific_dim - len(specific_features))
        feature_vector = type_onehot + specific_features
        sequence.append(np.array(feature_vector))

    # Process unmatched schedules
    node_names = {node.get('Name', '') for node in nodes}
    for sched in sched_data:
        name = sched.get('Name', '')
        if name and name not in node_names:
            type_onehot = [0.0, 0.0, 1.0]
            specific_features = []
            specific_features.extend(embed_categorical(name, 'node_names'))  # 10
            specific_features.extend([0.0] * 3)  # Access placeholder
            specific_features.extend([0.0] * 10)  # Op histogram placeholder
            specific_features.extend([0.0] * 10)  # Jacobian placeholder
            sched_vals = [
                sched.get('Details', {}).get('scheduling_feature', {}).get(k, 0)
                for k in ['inner_parallelism', 'outer_parallelism', 'vector_size', 'unrolled_loop_extent',
                          'points_computed_minimum', 'unique_bytes_read_per_realization', 'bytes_at_task']
            ]
            specific_features.extend(normalize_numerical(sched_vals, 7))  # 7
            # Ensure exact length
            specific_features = specific_features[:specific_dim] + [0.0] * (specific_dim - len(specific_features))
            feature_vector = type_onehot + specific_features
            sequence.append(np.array(feature_vector))

    # Pad or truncate
    max_len = 50
    if len(sequence) < max_len:
        sequence.extend([np.zeros(feature_dim)] * (max_len - len(sequence)))
    elif len(sequence) > max_len:
        sequence = sequence[:max_len]
    return np.array(sequence)

# Process a single Tiramisu JSON file
def process_tiramisu(tiramisu_data):
    prog = list(tiramisu_data.values())[0] if tiramisu_data else {}
    iterators = prog.get('program_annotation', {}).get('iterators', {})
    computations = prog.get('program_annotation', {}).get('computations', {})
    schedules = prog.get('schedules_list', [])

    sequences = []
    type_dim = 4  # Iterator, Computation, Execution, Fusion
    specific_dim = 40  # name(10) + bounds(2) + comp(5) + trans(20) + exec(3)
    feature_dim = type_dim + specific_dim  # 4 + 40 = 44

    for sched in schedules:
        sequence = []

        # Process iterators
        for it_name, it_data in iterators.items():
            type_onehot = [1.0, 0.0, 0.0, 0.0]
            specific_features = []
            specific_features.extend(embed_categorical(it_name, 'iterator_names', 10))  # 10
            bounds = [it_data.get('lower_bound', 0), it_data.get('upper_bound', 0)]
            bounds = [parse_number(str(b)) for b in bounds]
            specific_features.extend(normalize_numerical(bounds, 2))  # 2
            specific_features.extend([0.0] * 5)  # Comp placeholder
            specific_features.extend([0.0] * 20)  # Trans placeholder
            specific_features.extend([0.0] * 3)  # Exec placeholder
            # Ensure exact length
            specific_features = specific_features[:specific_dim] + [0.0] * (specific_dim - len(specific_features))
            feature_vector = type_onehot + specific_features
            sequence.append(np.array(feature_vector))

        # Process computations
        for comp_name, comp_data in computations.items():
            type_onehot = [0.0, 1.0, 0.0, 0.0]
            specific_features = []
            specific_features.extend(embed_categorical(comp_name, 'comp_names', 10))  # 10
            specific_features.extend([0.0] * 2)  # Bounds placeholder
            comp_features = [comp_data.get('absolute_order', 0) / 3.0,
                             1.0 if comp_data.get('comp_is_reduction', False) else 0.0,
                             len(comp_data.get('iterators', [])) / 4.0]
            specific_features.extend(normalize_numerical(comp_features, 5))  # 5
            trans = sched.get(comp_name, {})
            trans_features = []
            for t_type in vocab['transformation_types']:
                val = trans.get(t_type)
                if t_type == 'parallelized_dim' and val in vocab['iterator_names']:
                    trans_features.extend(embed_categorical(val, 'iterator_names', 4))
                elif t_type == 'unrolling_factor' and isinstance(val, (int, float)):
                    trans_features.append(val / 10.0)
                elif t_type in ['tiling', 'shiftings'] and isinstance(val, list):
                    vals = [parse_number(str(v)) for v in val[:5]]
                    vals += [0.0] * (5 - len(vals))
                    trans_features.extend(normalize_numerical(vals, 5))
                else:
                    trans_features.extend([0.0] * (4 if t_type == 'parallelized_dim' else 5 if t_type in ['tiling', 'shiftings'] else 1))
            specific_features.extend(trans_features[:20])  # 20
            specific_features.extend([0.0] * 3)  # Exec placeholder
            # Ensure exact length
            specific_features = specific_features[:specific_dim] + [0.0] * (specific_dim - len(specific_features))
            feature_vector = type_onehot + specific_features
            sequence.append(np.array(feature_vector))

        # Process execution time
        exec_times = sched.get('execution_times', [0.0])
        type_onehot = [0.0, 0.0, 1.0, 0.0]
        specific_features = []
        specific_features.extend([0.0] * 10)  # Name placeholder
        specific_features.extend([0.0] * 2)  # Bounds placeholder
        specific_features.extend([0.0] * 5)  # Comp placeholder
        specific_features.extend([0.0] * 20)  # Trans placeholder
        exec_stats = [np.mean(exec_times) / 0.01, np.std(exec_times) / 0.01, len(exec_times) / 10.0]
        specific_features.extend(normalize_numerical(exec_stats, 3))  # 3
        # Ensure exact length
        specific_features = specific_features[:specific_dim] + [0.0] * (specific_dim - len(specific_features))
        feature_vector = type_onehot + specific_features
        sequence.append(np.array(feature_vector))

        # Process fusions
        fusions = sched.get('fusions', [])
        if fusions:
            type_onehot = [0.0, 0.0, 0.0, 1.0]
            specific_features = []
            fusion_name = '_'.join(fusions[:2])
            specific_features.extend(embed_categorical(fusion_name, 'comp_names', 10))  # 10
            specific_features.extend([0.0] * 2)  # Bounds placeholder
            specific_features.extend([0.0] * 5)  # Comp placeholder
            specific_features.extend([0.0] * 20)  # Trans placeholder
            specific_features.extend([0.0] * 3)  # Exec placeholder
            # Ensure exact length
            specific_features = specific_features[:specific_dim] + [0.0] * (specific_dim - len(specific_features))
            feature_vector = type_onehot + specific_features
            sequence.append(np.array(feature_vector))

        # Pad or truncate
        max_len = 50
        if len(sequence) < max_len:
            sequence.extend([np.zeros(feature_dim)] * (max_len - len(sequence)))
        elif len(sequence) > max_len:
            sequence = sequence[:max_len]
        sequences.append(np.array(sequence))

    return sequences

# Process all Halide programs
def process_all_halide(halide_dir):
    all_sequences = []
    for subfolder in os.listdir(halide_dir):
        subfolder_path = os.path.join(halide_dir, subfolder)
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                if file.endswith('.json'):
                    file_path = os.path.join(subfolder_path, file)
                    with open(file_path, 'r') as f:
                        halide_data = json.load(f)
                    sequence = process_halide(halide_data)
                    all_sequences.append(sequence)
    return all_sequences

# Process all Tiramisu programs
def process_all_tiramisu(tiramisu_dir):
    all_sequences = []
    for file in os.listdir(tiramisu_dir):
        if file.endswith('.json'):
            file_path = os.path.join(tiramisu_dir, file)
            with open(file_path, 'r') as f:
                tiramisu_data = json.load(f)
            sequences = process_tiramisu(tiramisu_data)
            all_sequences.extend(sequences)
    return all_sequences

# Main execution
halide_dir = 'synthetic_data'
tiramisu_dir = 'Tiramisu'

halide_sequences = process_all_halide(halide_dir)
tiramisu_sequences = process_all_tiramisu(tiramisu_dir)

for i, seq in enumerate(halide_sequences):
    np.save(f'halide_data_{i}.npy', seq)
for i, seq in enumerate(tiramisu_sequences):
    np.save(f'tiramisu_data_{i}.npy', seq)

print("Halide Sequences Count:", len(halide_sequences))
if halide_sequences:
    print("Halide Sequence Shape:", halide_sequences[0].shape)
print("Tiramisu Sequences Count:", len(tiramisu_sequences))
if tiramisu_sequences:
    print("Tiramisu Sequence Shape:", tiramisu_sequences[0].shape)
