import json
import os
from pathlib import Path
import uuid

def convert_tree_to_function_graph(tree_data):
    output = {"without_extern": {"nodes": [], "edges": [], "global_features": {}}}
    
    # Extract global features from the node with id -1
    global_node = next((node for node in tree_data['children'] if node['id'] == -1), None)
    if global_node:
        output['without_extern']['global_features'] = {
            'cache_hits': global_node.get('cache_hits', 0),
            'cache_misses': global_node.get('cache_misses', 0),
            'execution_time_ms': global_node.get('execution_time_ms', 0.0)
        }
    
    # Collect all nodes except Root and Global Features (id < 0)
    nodes = [node for node in tree_data['children'] if node['id'] >= 0]
    
    # Process nodes
    for node in nodes:
        node_id = node['id']
        name = node['name']
        
        # Determine if node is input or output
        is_input = len(node.get('children', [])) == 0 and 'update' not in name.lower()
        is_output = any(child.get('target_id') == node_id for n in nodes for child in n.get('children', []))
        
        # Map memory patterns
        memory_patterns = node.get('memory_patterns', {})
        pipeline_features = {
            'memory_access_patterns': {
                'Float': {
                    'Broadcast': memory_patterns.get('Broadcast', [0, 0, 0, 0]),
                    'Pointwise': memory_patterns.get('Pointwise', [0, 0, 0, 0]),
                    'Slice': memory_patterns.get('Slice', [0, 0, 0, 0]),
                    'Transpose': memory_patterns.get('Transpose', [0, 0, 0, 0])
                },
                'UInt32': {
                    'Broadcast': [0, 0, 0, 0],
                    'Pointwise': [0, 0, 0, 0],
                    'Slice': [0, 0, 0, 0],
                    'Transpose': [0, 0, 0, 0]
                }
            },
            'op_histogram': {
                'Float': node.get('op_histogram', {}),
                'UInt32': {k: 0 for k in node.get('op_histogram', {})}
            }
        }
        
        # Map scheduling features
        sched = node.get('scheduling', {})
        schedule_features = {
            'allocation_bytes_read_per_realization': sched.get('allocation_bytes_read_per_realization', 0),
            'bytes_at_production': sched.get('bytes_at_production', 0),
            'bytes_at_realization': sched.get('bytes_at_realization', 0),
            'bytes_at_root': sched.get('bytes_at_root', 0),
            'bytes_at_task': sched.get('bytes_at_task', 0),
            'inlined_calls': sched.get('inlined_calls', 0),
            'inner_parallelism': sched.get('inner_parallelism', 0),
            'innermost_bytes_at_production': sched.get('innermost_bytes_at_production', 0),
            'innermost_bytes_at_realization': sched.get('innermost_bytes_at_realization', 0),
            'innermost_bytes_at_root': sched.get('innermost_bytes_at_root', 0),
            'innermost_bytes_at_task': sched.get('innermost_bytes_at_task', 0),
            'innermost_loop_extent': sched.get('innermost_loop_extent', 0),
            'innermost_pure_loop_extent': sched.get('innermost_pure_loop_extent', 0),
            'native_vector_size': sched.get('native_vector_size', 0),
            'num_productions': sched.get('num_productions', 0),
            'num_realizations': sched.get('num_realizations', 0),
            'num_scalars': sched.get('num_scalars', 0),
            'num_vectors': sched.get('num_vectors', 0),
            'outer_parallelism': sched.get('outer_parallelism', 0),
            'points_computed_minimum': sched.get('points_computed_minimum', 0),
            'points_computed_per_production': sched.get('points_computed_per_production', 0),
            'points_computed_per_realization': sched.get('points_computed_per_realization', 0),
            'points_computed_total': sched.get('points_computed_total', 0),
            'scalar_loads_per_scalar': sched.get('scalar_loads_per_scalar', 0),
            'scalar_loads_per_vector': sched.get('scalar_loads_per_vector', 0),
            'unique_bytes_read_per_realization': sched.get('unique_bytes_read_per_realization', 0),
            'unique_bytes_read_per_task': sched.get('unique_bytes_read_per_task', 0),
            'unique_bytes_read_per_vector': sched.get('unique_bytes_read_per_vector', 0),
            'unique_lines_read_per_realization': sched.get('unique_lines_read_per_realization', 0),
            'unique_lines_read_per_task': sched.get('unique_lines_read_per_task', 0),
            'unique_lines_read_per_vector': sched.get('unique_lines_read_per_vector', 0),
            'unrolled_loop_extent': sched.get('unrolled_loop_extent', 0),
            'vector_loads_per_vector': sched.get('vector_loads_per_vector', 0),
            'vector_size': sched.get('vector_size', 0),
            'working_set': sched.get('working_set', 0),
            'working_set_at_production': sched.get('working_set_at_production', 0),
            'working_set_at_realization': sched.get('working_set_at_realization', 0),
            'working_set_at_root': sched.get('working_set_at_root', 0),
            'working_set_at_task': sched.get('working_set_at_task', 0)
        }
        
        # Infer loops from footprint or assume defaults
        loops = [
            {'var': 'x', 'min': f"{name}._0.min", 'max': f"{name}._0.max"},
            {'var': 'y', 'min': f"{name}._1.min", 'max': f"{name}._1.max"}
        ]
        if 'update' in name.lower():
            loops.insert(0, {'var': 'r4$x', 'min': '0', 'max': '199'})
        
        # Construct node
        node_entry = {
            'id': node_id,
            'name': name,
            'input': is_input,
            'output': is_output,
            'pointwise': any(p[3] == 1 for p in memory_patterns.values()),
            'boundary_condition': False,
            'wrapper': 'update' in name.lower(),
            'region_computed': [
                {'min': f"{name}._0.min", 'max': f"{name}._0.max"},
                {'min': f"{name}._1.min", 'max': f"{name}._1.max"}
            ],
            'region_required': [
                {'min': f"{name}._0.min", 'max': f"{name}._0.max"},
                {'min': f"{name}._1.min", 'max': f"{name}._1.max"}
            ],
            'stages': [{
                'index': 0,
                'loops': loops,
                'pipeline_features': pipeline_features,
                'schedule_features': schedule_features
            }]
        }
        
        output['without_extern']['nodes'].append(node_entry)
    
    # Process edges
    for node in nodes:
        for child in node.get('children', []):
            target_id = child['target_id']
            target_name = child['target_name']
            source_name = node['name']
            source_id = node['id']
            
            # Map footprint to bounds
            footprint = child.get('footprint', {})
            bounds = [
                {'min': footprint.get('Min 0', f"{target_name}._0.min"), 'max': footprint.get('Max 0', f"{target_name}._0.max")},
                {'min': footprint.get('Min 1', f"{target_name}._1.min"), 'max': footprint.get('Max 1', f"{target_name}._1.max")}
            ]
            
            # Map load_jacobian
            load_jacobian = child.get('load_jacobian', [[1, 0], [0, 1]])
            if load_jacobian and any('-' in row for row in load_jacobian):
                load_jacobian = [[1, 0], [0, 1]]  # Default to identity if invalid
            
            edge = {
                'source': source_name,
                'source_id': source_id,
                'target': target_name,
                'target_id': target_id,
                'bounds': bounds,
                'load_jacobians': [{
                    'count': 1,
                    'matrix': load_jacobian[:2]  # Ensure 2x2 matrix
                }]
            }
            
            output['without_extern']['edges'].append(edge)
    
    return output

def process_directory(input_dir='Tree_Output', output_dir='Graph_Output'):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True)
    
    # Walk through all subdirectories in input_dir
    for root, dirs, files in os.walk(input_path):
        if 'tree_representation.json' in files:
            input_file = Path(root) / 'tree_representation.json'
            
            # Compute relative path to maintain folder structure
            rel_path = input_file.parent.relative_to(input_path)
            output_subdir = output_path / rel_path
            
            # Create corresponding output subdirectory
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # Read and process the input file
            try:
                with open(input_file, 'r') as f:
                    tree_data = json.load(f)
                
                # Convert to function graph
                result = convert_tree_to_function_graph(tree_data)
                
                # Save output to corresponding location
                output_file = output_subdir / 'converted_function_graph.json'
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=4)
                
                print(f"Processed: {input_file} -> {output_file}")
            
            except Exception as e:
                print(f"Error processing {input_file}: {str(e)}")

if __name__ == '__main__':
    process_directory()
