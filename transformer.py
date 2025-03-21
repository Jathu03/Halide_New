import json
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# Vocabulary for categorical features
vocab = {
    'op_types': ['Constant', 'Cast', 'Variable', 'Param', 'Add', 'Sub', 'Mul', 'Div', 'Min', 'Max', 'EQ', 'NE', 'LT', 'LE', 'And', 'Or', 'Not', 'Select', 'ImageCall', 'FuncCall', 'SelfCall', 'ExternCall', 'Let'],
    'iterator_names': ['i0', 'i1', 'i2', 'i3'],
    'transformation_types': ['shiftings', 'tiling', 'unrolling_factor', 'parallelized_dim'],
    'access_types': ['Pointwise'],
    'comp_names': ['comp00', 'comp01', 'comp02']
}
vocab_sizes = {k: len(v) for k, v in vocab.items()}

# Helper function to embed categorical values as one-hot vectors
def embed_categorical(value, vocab_key):
    idx = vocab[vocab_key].index(value) if value in vocab[vocab_key] else -1
    one_hot = np.zeros(vocab_sizes[vocab_key])
    if idx >= 0:
        one_hot[idx] = 1
    return one_hot

# Helper function to normalize numerical values
def normalize_numerical(values):
    scaler = MinMaxScaler()
    values = np.array(values).reshape(-1, 1)
    return scaler.fit_transform(values).flatten()

# Helper function to parse numbers, handling fractions and invalid strings
def parse_number(x):
    try:
        if '/' in x:
            return float(eval(x))  # Evaluate fractions like '1/16'
        return float(x)  # Convert regular numbers
    except (ValueError, SyntaxError, NameError):
        return 0.0  # Return 0.0 for invalid strings like '_'

# Process a single Halide JSON file
def process_halide(halide_data):
    edges = halide_data['programming_details']['Edges']
    nodes = halide_data['programming_details']['Nodes']
    sched_data = halide_data['scheduling_data']

    sequence = []
    type_dim = 2
    specific_dim = 21
    feature_dim = type_dim + specific_dim  # 2 + 21 = 23

    for node in nodes:
        type_onehot = [1.0, 0.0]  # Node type
        node_features = []
        name = node['Name']
        details = node['Details']
        access_pattern = details['Memory access patterns'][0].split(':')[1].strip()
        node_features.extend(embed_categorical(access_pattern.split()[0], 'access_types'))
        op_hist = details['Op histogram']
        op_counts = [int(line.split(':')[1].strip()) for line in op_hist[:10]]
        node_features.extend(normalize_numerical(op_counts))
        sched = next((s.get('Details', {}).get('scheduling_feature', {}) for s in sched_data if 'Name' in s and s['Name'] == name), {})
        sched_vals = [
            sched.get('inner_parallelism', 0),
            sched.get('outer_parallelism', 0),
            sched.get('vector_size', 0),
            sched.get('unrolled_loop_extent', 0),
            sched.get('points_computed_total', 0),
            sched.get('unique_bytes_read_per_realization', 0),
            sched.get('bytes_at_production', 0),
            sched.get('bytes_at_realization', 0),
            sched.get('bytes_at_root', 0),
            sched.get('bytes_at_task', 0)
        ]
        node_features.extend(normalize_numerical(sched_vals))
        feature_vector = type_onehot + node_features
        sequence.append(np.array(feature_vector))

    for edge in edges:
        type_onehot = [0.0, 1.0]  # Edge type
        edge_features = []
        jacobian_str = ' '.join(edge['Details']['Load Jacobians'])
        jacobian = [parse_number(x) for x in jacobian_str.split()]
        if not jacobian:  # If jacobian is empty
            jacobian = [0.0] * 9
        edge_features.extend(normalize_numerical(jacobian[:9]))
        specific_features = edge_features + [0.0] * (specific_dim - len(edge_features))
        feature_vector = type_onehot + specific_features
        sequence.append(np.array(feature_vector))

    max_len = 50
    if len(sequence) < max_len:
        padding = [np.zeros(feature_dim)] * (max_len - len(sequence))
        sequence.extend(padding)
    elif len(sequence) > max_len:
        sequence = sequence[:max_len]

    return np.array(sequence)

# Process a single Tiramisu JSON file
def process_tiramisu(tiramisu_data):
    prog = list(tiramisu_data.values())[0]
    iterators = prog['program_annotation']['iterators']
    computations = prog['program_annotation']['computations']
    schedules = prog['schedules_list']

    sequences = []
    type_dim = 3
    specific_dim = 18
    feature_dim = type_dim + specific_dim  # 3 + 18 = 21

    for sched in schedules:
        sequence = []

        for it_name, it_data in iterators.items():
            type_onehot = [1.0, 0.0, 0.0]  # Iterator type
            it_features = list(embed_categorical(it_name, 'iterator_names'))
            bounds = [it_data['lower_bound'], it_data['upper_bound']]
            bounds = [0 if isinstance(b, str) else b for b in bounds]
            it_features.extend(normalize_numerical(bounds))
            specific_features = it_features + [0.0] * (specific_dim - len(it_features))
            feature_vector = type_onehot + specific_features
            sequence.append(np.array(feature_vector))

        for comp_name, comp_data in computations.items():
            type_onehot = [0.0, 1.0, 0.0]  # Computation type
            comp_features = list(embed_categorical(comp_name, 'comp_names'))
            comp_features.append(comp_data['absolute_order'] / 3.0)
            comp_features.append(1.0 if comp_data['comp_is_reduction'] else 0.0)
            trans = sched.get(comp_name, {})
            trans_features = []
            if 'parallelized_dim' in trans and trans['parallelized_dim'] in vocab['iterator_names']:
                idx = vocab['iterator_names'].index(trans['parallelized_dim'])
                parallel_onehot = [0.0] * vocab_sizes['iterator_names']
                parallel_onehot[idx] = 1.0
                trans_features.extend(parallel_onehot)
            else:
                trans_features.extend([0.0] * vocab_sizes['iterator_names'])
            if 'unrolling_factor' in trans and isinstance(trans['unrolling_factor'], (int, float)):
                trans_features.append(trans['unrolling_factor'] / 10.0)
            else:
                trans_features.append(0.0)
            if 'tiling' in trans and isinstance(trans['tiling'], list):
                tiling_vals = [v / 10.0 for v in trans['tiling'][:4]]
                tiling_vals += [0.0] * (4 - len(tiling_vals))
                trans_features.extend(tiling_vals)
            else:
                trans_features.extend([0.0] * 4)
            if 'shiftings' in trans and isinstance(trans['shiftings'], list):
                shift_vals = [v / 10.0 for v in trans['shiftings'][:4]]
                shift_vals += [0.0] * (4 - len(shift_vals))
                trans_features.extend(shift_vals)
            else:
                trans_features.extend([0.0] * 4)
            specific_features = comp_features + trans_features
            feature_vector = type_onehot + specific_features
            sequence.append(np.array(feature_vector))

        exec_times = sched['execution_times']
        mean_exec = np.mean(exec_times) / 0.01
        type_onehot = [0.0, 0.0, 1.0]  # Execution time type
        specific_features = [mean_exec] + [0.0] * (specific_dim - 1)
        feature_vector = type_onehot + specific_features
        sequence.append(np.array(feature_vector))

        max_len = 50
        if len(sequence) < max_len:
            padding = [np.zeros(feature_dim)] * (max_len - len(sequence))
            sequence.extend(padding)
        elif len(sequence) > max_len:
            sequence = sequence[:max_len]

        sequences.append(np.array(sequence))

    return sequences

# Process all Halide programs across subfolders
def process_all_halide(halide_dir):
    all_sequences = []
    first_file = None
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
                    if first_file is None:  # Store the first file processed
                        first_file = file_path
                        first_sequence = sequence
    return all_sequences, first_file, first_sequence if first_file else None

# Process all Tiramisu programs in the folder
def process_all_tiramisu(tiramisu_dir):
    all_sequences = []
    first_file = None
    for file in os.listdir(tiramisu_dir):
        if file.endswith('.json'):
            file_path = os.path.join(tiramisu_dir, file)
            with open(file_path, 'r') as f:
                tiramisu_data = json.load(f)
            sequences = process_tiramisu(tiramisu_data)
            all_sequences.extend(sequences)
            if first_file is None:  # Store the first file processed
                first_file = file_path
                first_sequence = sequences[0]  # Take the first schedule
    return all_sequences, first_file, first_sequence if first_file else None

# Main execution
halide_dir = 'synthetic_data'
tiramisu_dir = 'Tiramisu'

# Generate sequences and get first representations
halide_sequences, halide_first_file, halide_first_seq = process_all_halide(halide_dir)
tiramisu_sequences, tiramisu_first_file, tiramisu_first_seq = process_all_tiramisu(tiramisu_dir)

# Save sequences as NumPy files
for i, seq in enumerate(halide_sequences):
    np.save(f'halide_data_{i}.npy', seq)
for i, seq in enumerate(tiramisu_sequences):
    np.save(f'tiramisu_data_{i}.npy', seq)

# Print verification
print("Halide Sequences Count:", len(halide_sequences))
if halide_sequences:
    print("Halide Sequence Shape:", halide_sequences[0].shape)
print("Tiramisu Sequences Count:", len(tiramisu_sequences))
if tiramisu_sequences:
    print("Tiramisu Sequence Shape:", tiramisu_sequences[0].shape)

# Display representations for one file each
print("\n=== Halide Representation (First File) ===")
if halide_first_file:
    print(f"File: {halide_first_file}")
    print("Shape:", halide_first_seq.shape)
    print("First 5 timesteps (subset of features):")
    np.set_printoptions(precision=3, suppress=True)  # Format for readability
    print(halide_first_seq[:5, :5])  # Show first 5 timesteps, first 5 features
    print("...")
else:
    print("No Halide files processed.")

print("\n=== Tiramisu Representation (First File, First Schedule) ===")
if tiramisu_first_file:
    print(f"File: {tiramisu_first_file}")
    print("Shape:", tiramisu_first_seq.shape)
    print("First 5 timesteps (subset of features):")
    np.set_printoptions(precision=3, suppress=True)
    print(tiramisu_first_seq[:5, :5])  # Show first 5 timesteps, first 5 features
    print("...")
else:
    print("No Tiramisu files processed.")
