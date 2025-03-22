import json
import numpy as np
import torch
from pathlib import Path
import os
from tqdm import tqdm

# Constants from the document
MAX_NUM_TRANSFORMATIONS = 4
MAX_TAGS = 16
MAX_DEPTH = 5
MAX_EXPR_LEN = 66
MAX_ACCESSES = 15

# Simplified version of isl_to_write_matrix (assuming access relations are matrices or parsed similarly)
def isl_to_write_matrix(isl_map):
    # Placeholder: Assuming isl_map is already a matrix or parsed into one
    # In practice, you'd parse the ISL string as in the document
    return np.array(isl_map) if isinstance(isl_map, list) else np.zeros((MAX_DEPTH, MAX_DEPTH + 1))

def pad_access_matrix(access_matrix):
    access_matrix = np.array(access_matrix)
    if access_matrix.size == 0:
        return np.zeros((MAX_DEPTH + 1, MAX_DEPTH + 2))
    padded = np.zeros((MAX_DEPTH + 1, MAX_DEPTH + 2))
    rows, cols = min(access_matrix.shape[0], MAX_DEPTH + 1), min(access_matrix.shape[1], MAX_DEPTH + 1)
    padded[:rows, :cols] = access_matrix[:rows, :cols]
    return padded

# Simplified expression representation (one-hot encoding)
def get_expr_repr(expr_type, comp_type):
    expr_map = {"add": 0, "sub": 1, "mul": 2, "div": 3, "sqrt": 4, "min": 5, "max": 6, "unknown": 7}
    type_map = {"int32": 0, "float32": 1, "float64": 2}
    expr_vec = [0] * 8
    type_vec = [0] * 3
    expr_vec[expr_map.get(expr_type, 7)] = 1
    type_vec[type_map.get(comp_type, 1)] = 1
    return expr_vec + type_vec

def get_tree_expr_repr(node, comp_type):
    expr_tensor = []
    if isinstance(node, dict) and "children" in node and node["children"]:
        for child in node["children"]:
            expr_tensor.extend(get_tree_expr_repr(child, comp_type))
    expr_tensor.append(get_expr_repr(node.get("expr_type", "unknown") if isinstance(node, dict) else "unknown", comp_type))
    padded = expr_tensor + [[0] * 11] * (MAX_EXPR_LEN - len(expr_tensor))
    return padded[:MAX_EXPR_LEN]

# Simplified transformation tags (assuming transformations_list is a list of tags)
def get_padded_transformation_tags(schedule_dict):
    tags = []
    for t in schedule_dict.get("transformations_list", []):
        tags.extend(t if isinstance(t, list) else [0] * MAX_TAGS)
    return tags + [0] * (MAX_NUM_TRANSFORMATIONS * MAX_TAGS - len(tags))

# Data representation creation
def create_data_representation(tiramisu_dir="Tiramisu"):
    dataset = []
    for json_file in tqdm(list(Path(tiramisu_dir).glob("*.json"))):
        with open(json_file, "r") as f:
            data = json.load(f)
        
        function_key = list(data.keys())[0]
        program_json = data[function_key]["program_annotation"]
        schedules = data[function_key]["schedules_list"]

        # Computations
        comps_dict = program_json["computations"]
        ordered_comps = sorted(comps_dict.keys(), key=lambda x: comps_dict[x]["absolute_order"])
        
        for sched in schedules:
            if not sched.get("execution_times") or min(sched["execution_times"]) <= 0:
                continue
            
            comps_repr = []
            loops_repr = []
            expr_repr = []
            
            # Computation representation
            for comp_idx, comp_name in enumerate(ordered_comps):
                comp_dict = comps_dict[comp_name]
                sched_dict = sched.get(comp_name, {})
                
                # Basic features
                comp_features = [int(comp_dict["comp_is_reduction"])]
                
                # Iterators (loop nest)
                iterators = comp_dict["iterators"]
                iter_repr = []
                for i, iter_name in enumerate(iterators[:MAX_DEPTH]):
                    l_code = f"C{comp_idx}-L{i}"
                    # Ensure shiftings and fusions are lists, defaulting to [] if None
                    shiftings = sched_dict.get("shiftings") if sched_dict.get("shiftings") is not None else []
                    fusions = sched.get("fusions") if sched.get("fusions") is not None else []
                    iter_repr.extend([
                        int(iter_name == sched_dict.get("parallelized_dim", "")),  # Parallelized
                        int(sched_dict.get("tiling", {}).get("tiling_dims", []).count(iter_name) > 0),  # Tiled
                        int(sched_dict.get("tiling", {}).get("tiling_factors", [0])[i]) if i < len(sched_dict.get("tiling", {}).get("tiling_factors", [])) else 0,  # TileFactor
                        int(i in [f[2] for f in fusions if comp_name in f]),  # Fused
                        int(any(iter_name.startswith(s[0]) for s in shiftings)),  # Shifted
                        next((s[1] for s in shiftings if iter_name.startswith(s[0])), 0)  # ShiftFactor
                    ])
                iter_repr += [0] * 6 * (MAX_DEPTH - len(iterators))  # Pad
                
                # Handle unrolling_factor, ensuring None is treated as 0
                unrolling_factor = sched_dict.get("unrolling_factor")
                unrolling_factor = 0 if unrolling_factor is None else unrolling_factor
                iter_repr.extend([
                    int(bool(unrolling_factor)),  # Unrolled
                    int(unrolling_factor)  # UnrollFactor
                ])
                
                # Transformation tags
                tags = get_padded_transformation_tags(sched_dict)
                iter_repr.extend(tags)
                
                # Access matrices (simplified)
                write_mat = pad_access_matrix(isl_to_write_matrix(comp_dict["write_access_relation"]))
                comp_features.extend([comp_dict["write_buffer_id"] + 1] + write_mat.flatten().tolist())
                
                read_accesses = []
                for acc in comp_dict["accesses"][:MAX_ACCESSES]:
                    read_mat = pad_access_matrix(acc["access_matrix"])
                    read_accesses.extend([int(acc["access_is_reduction"]), acc["buffer_id"] + 1] + read_mat.flatten().tolist())
                read_accesses += [0] * ((MAX_DEPTH + 1) * (MAX_DEPTH + 2) + 2) * (MAX_ACCESSES - len(comp_dict["accesses"]))
                comp_features.extend(read_accesses)
                
                comps_repr.append(comp_features)
                
                # Expression representation
                expr_repr.append(get_tree_expr_repr(comp_dict["expression_representation"], comp_dict["data_type"]))
            
            # Loops representation (simplified, assuming iterators are global)
            loops_dict = program_json["iterators"]
            loops_features = []
            for loop_name in loops_dict.keys():
                sched_comp = sched.get(ordered_comps[0], {})
                # Ensure shiftings and fusions are lists, defaulting to [] if None
                shiftings = sched_comp.get("shiftings") if sched_comp.get("shiftings") is not None else []
                fusions = sched.get("fusions") if sched.get("fusions") is not None else []
                # Handle unrolling_factor for loops representation
                unrolling_factor = sched_comp.get("unrolling_factor")
                unrolling_factor = 0 if unrolling_factor is None else unrolling_factor
                l_repr = [
                    int(loop_name == sched_comp.get("parallelized_dim", "")),
                    int(sched_comp.get("tiling", {}).get("tiling_dims", []).count(loop_name) > 0),
                    int(sched_comp.get("tiling", {}).get("tiling_factors", [0])[0]),
                    int(any(loop_name in comps_dict[c]["iterators"][f[2]] for f in fusions for c in f[:2])),
                    int(bool(unrolling_factor)),
                    int(unrolling_factor),
                    int(any(loop_name.startswith(s[0]) for s in shiftings)),
                    next((s[1] for s in shiftings if loop_name.startswith(s[0])), 0)
                ]
                loops_features.append(l_repr)
            
            # Execution time
            exec_time = np.mean(sched["execution_times"])
            
            dataset.append({
                "comps_tensor": torch.tensor(comps_repr, dtype=torch.float32),
                "loops_tensor": torch.tensor(loops_features, dtype=torch.float32),
                "expr_tensor": torch.tensor(expr_repr, dtype=torch.float32),
                "exec_time": exec_time
            })
    
    torch.save(dataset, "tiramisu_dataset.pt")
    print(f"Dataset saved to tiramisu_dataset.pt with {len(dataset)} samples")

if __name__ == "__main__":
    create_data_representation()
