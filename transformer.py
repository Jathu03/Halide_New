import json
import numpy as np
import os
from collections import defaultdict

# Directory containing Tiramisu JSON files
TIRAMISU_DIR = './Tiramisu'  # Adjust this path if needed

# Function to parse expression tree into a string
def parse_expression(expr):
    if not isinstance(expr, dict) or 'children' not in expr or not expr['children']:
        return str(expr.get('expr_type', 'unknown')) if isinstance(expr, dict) else str(expr)
    children = [parse_expression(child) for child in expr['children']]
    return f"{expr['expr_type']}({','.join(children)})"

# Build base program graph
def build_program_graph(program_data):
    graph = {
        'nodes': {},
        'edges': [],
        'attributes': {}
    }
    
    # Add iterator nodes
    iterators = program_data.get('iterators', {})
    for it_name, it_data in iterators.items():
        lower_bound = it_data.get('lower_bound', 'unknown')
        upper_bound = it_data.get('upper_bound', 'unknown')
        if isinstance(lower_bound, int) and isinstance(upper_bound, int):
            range_str = f"{lower_bound}-{upper_bound}"
        else:
            range_str = f"{str(lower_bound)}-{str(upper_bound)}"
        graph['nodes'][it_name] = 'iterator'
        graph['attributes'][it_name] = {
            'type': 'iterator',
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'range': range_str,
            'parent': it_data.get('parent_iterator', None)
        }
        if it_data.get('parent_iterator'):
            graph['edges'].append((it_data['parent_iterator'], it_name, 'nesting'))
        computations_list = it_data.get('computations_list', [])
        for comp in computations_list:
            graph['edges'].append((it_name, comp, 'contains'))

    # Add computation nodes
    computations = program_data.get('computations', {})
    for comp_name, comp_data in computations.items():
        graph['nodes'][comp_name] = 'computation'
        accesses = [str(access.get('access_matrix', 'unknown')) for access in comp_data.get('accesses', [])]
        expr_str = parse_expression(comp_data.get('expression_representation', {}))
        graph['attributes'][comp_name] = {
            'type': 'computation',
            'is_reduction': comp_data.get('comp_is_reduction', False),
            'write_access': comp_data.get('write_access_relation', 'unknown'),
            'write_buffer_id': comp_data.get('write_buffer_id', 'unknown'),
            'data_type': comp_data.get('data_type', 'unknown'),
            'accesses': accesses,
            'expression': expr_str,
            'iterators': comp_data.get('iterators', [])
        }
    
    return graph

# Apply schedule transformations to the graph
def apply_schedule_to_graph(base_graph, schedule):
    graph = {
        'nodes': base_graph['nodes'].copy(),
        'edges': base_graph['edges'].copy(),
        'attributes': {k: v.copy() for k, v in base_graph['attributes'].items()},
        'tree_structure': schedule.get('tree_structure', 'unknown')
    }
    
    # Process fusions
    fusions = schedule.get('fusions', [])
    if fusions is not None:
        for fusion in fusions:
            if isinstance(fusion, list) and len(fusion) >= 3:
                comps, level = fusion[0:2], fusion[2]
                if isinstance(comps, list):
                    for comp in comps:
                        graph['attributes'][comp]['fused'] = True
                        graph['attributes'][comp]['fusion_level'] = f"L{level}"
            else:
                print(f"Skipping invalid fusion format: {fusion}")
    
    # Process transformations
    for comp_name, comp_data in schedule.items():
        if comp_name in ['fusions', 'sched_str', 'tree_structure', 'legality_check', 
                        'exploration_method', 'execution_times']:
            continue
        attrs = graph['attributes'].get(comp_name, {})
        if isinstance(comp_data, dict):
            if comp_data.get('shiftings'):
                attrs['shiftings'] = comp_data['shiftings']
            if comp_data.get('tiling'):
                attrs['tiling'] = comp_data['tiling']
            if comp_data.get('unrolling_factor'):
                attrs['unrolling_factor'] = comp_data['unrolling_factor']
            if comp_data.get('parallelized_dim'):
                attrs['parallelized_dim'] = comp_data['parallelized_dim']
            if comp_data.get('transformations_list'):
                attrs['transformations'] = comp_data['transformations_list']
            graph['attributes'][comp_name] = attrs
    
    graph['schedule_str'] = schedule.get('sched_str', 'unknown')
    return graph

# Main processing: Iterate through all JSON files in the Tiramisu folder
dataset = []
if not os.path.exists(TIRAMISU_DIR):
    print(f"Error: Directory '{TIRAMISU_DIR}' not found.")
    exit(1)

json_files = [f for f in os.listdir(TIRAMISU_DIR) if f.endswith('.json')]
if not json_files:
    print(f"Error: No JSON files found in '{TIRAMISU_DIR}'.")
    exit(1)

for json_file in json_files:
    file_path = os.path.join(TIRAMISU_DIR, json_file)
    print(f"Processing {file_path}...")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract function key dynamically
        if not data or not isinstance(data, dict):
            print(f"Skipping {file_path}: Empty or invalid JSON")
            continue
        function_key = list(data.keys())[0]  # Assume first key is the function name
        program_data = data[function_key].get('program_annotation', {})
        schedules = data[function_key].get('schedules_list', [])

        if not program_data or not schedules:
            print(f"Skipping {file_path}: Missing 'program_annotation' or 'schedules_list'")
            continue

        # Build base graph for this program
        base_graph = build_program_graph(program_data)

        # Process each schedule
        for sched in schedules:
            if not isinstance(sched, dict) or 'execution_times' not in sched:
                print(f"Skipping schedule in {file_path}: Invalid schedule format or missing execution_times")
                continue
            sched_graph = apply_schedule_to_graph(base_graph, sched)
            exec_times = sched['execution_times']
            if not exec_times or not isinstance(exec_times, list):
                print(f"Skipping schedule in {file_path}: Invalid execution_times")
                continue
            avg_exec_time = np.mean(exec_times)
            dataset.append({
                'program_file': json_file,
                'graph': sched_graph,
                'avg_execution_time': avg_exec_time
            })
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Save dataset as JSON
output_file = 'tiramisu_graph_dataset.json'
with open(output_file, 'w') as f:
    json.dump(dataset, f, indent=2)
print(f"Graph dataset saved to '{output_file}'")

# Print summary
print(f"\nProcessed {len(json_files)} files, {len(dataset)} schedule graphs.")
if dataset:
    print("\nSample Graph Structure (first entry):")
    print(json.dumps(dataset[0]['graph'], indent=2)[:1000], "... (truncated)")
    print(f"Avg Execution Time: {dataset[0]['avg_execution_time']}")
