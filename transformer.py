import json
import numpy as np
from collections import defaultdict

# Load the JSON file
with open('data.json', 'r') as f:
    data = json.load(f)

# Function to parse expression tree into a string
def parse_expression(expr):
    if 'children' not in expr or not expr['children']:
        return expr['expr_type']
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
    iterators = program_data['iterators']
    for it_name, it_data in iterators.items():
        lower_bound = it_data['lower_bound'] if isinstance(it_data['lower_bound'], int) else str(it_data['lower_bound'])
        upper_bound = it_data['upper_bound'] if isinstance(it_data['upper_bound'], int) else str(it_data['upper_bound'])
        graph['nodes'][it_name] = 'iterator'
        graph['attributes'][it_name] = {
            'type': 'iterator',
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'range': f"{lower_bound}-{upper_bound}",
            'parent': it_data.get('parent_iterator', None)
        }
        if it_data.get('parent_iterator'):
            graph['edges'].append((it_data['parent_iterator'], it_name, 'nesting'))
        for comp in it_data['computations_list']:
            graph['edges'].append((it_name, comp, 'contains'))

    # Add computation nodes
    computations = program_data['computations']
    for comp_name, comp_data in computations.items():
        graph['nodes'][comp_name] = 'computation'
        accesses = [str(access['access_matrix']) for access in comp_data['accesses']]
        expr_str = parse_expression(comp_data['expression_representation'])
        graph['attributes'][comp_name] = {
            'type': 'computation',
            'is_reduction': comp_data['comp_is_reduction'],
            'write_access': comp_data['write_access_relation'],
            'write_buffer_id': comp_data['write_buffer_id'],
            'data_type': comp_data['data_type'],
            'accesses': accesses,  # Keep as list for structure
            'expression': expr_str,
            'iterators': comp_data['iterators']
        }
    
    return graph

# Apply schedule transformations to the graph
def apply_schedule_to_graph(base_graph, schedule):
    graph = {
        'nodes': base_graph['nodes'].copy(),
        'edges': base_graph['edges'].copy(),
        'attributes': {k: v.copy() for k, v in base_graph['attributes'].items()},
        'tree_structure': schedule['tree_structure']  # Include execution flow
    }
    
    # Process fusions
    fusions = schedule.get('fusions', [])
    for fusion in fusions:
        comps, level = fusion[0:2], fusion[2]
        for comp in comps:
            graph['attributes'][comp]['fused'] = True
            graph['attributes'][comp]['fusion_level'] = f"L{level}"
    
    # Process transformations
    for comp_name, comp_data in schedule.items():
        if comp_name in ['fusions', 'sched_str', 'tree_structure', 'legality_check', 
                        'exploration_method', 'execution_times']:
            continue
        attrs = graph['attributes'].get(comp_name, {})
        if comp_data['shiftings']:
            attrs['shiftings'] = comp_data['shiftings']
        if comp_data['tiling']:
            attrs['tiling'] = comp_data['tiling']
        if comp_data['unrolling_factor']:
            attrs['unrolling_factor'] = comp_data['unrolling_factor']
        if comp_data['parallelized_dim']:
            attrs['parallelized_dim'] = comp_data['parallelized_dim']
        if comp_data['transformations_list']:
            attrs['transformations'] = comp_data['transformations_list']
        graph['attributes'][comp_name] = attrs
    
    graph['schedule_str'] = schedule['sched_str']
    return graph

# Main processing
program_data = data['function003306']['program_annotation']
schedules = data['function003306']['schedules_list']

# Build base graph
base_graph = build_program_graph(program_data)

# Prepare dataset as list of graphs
dataset = []
for sched in schedules:
    sched_graph = apply_schedule_to_graph(base_graph, sched)
    exec_times = sched['execution_times']
    avg_exec_time = np.mean(exec_times)
    dataset.append({
        'graph': sched_graph,
        'avg_execution_time': avg_exec_time
    })

# Save dataset as JSON (preserving structure)
with open('tiramisu_graph_dataset.json', 'w') as f:
    json.dump(dataset, f, indent=2)
print("Graph dataset saved to 'tiramisu_graph_dataset.json'")

# Example: Print first graph’s structure
print("\nSample Graph Structure:")
print(json.dumps(dataset[0]['graph'], indent=2)[:1000], "... (truncated)")
print(f"Avg Execution Time: {dataset[0]['avg_execution_time']}")
