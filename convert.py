import os
import json

def process_function_data(data, function_id, output_dir):
    """Process a single function's data and create a separate file with program and schedule details."""
    func_data = data.get(function_id, {})
    
    # Extract program details
    program_details = {
        "function_id": function_id,
        "filename": func_data.get("filename", ""),
        "node_name": func_data.get("node_name", ""),
        "parameters": func_data.get("parameters", {}),
        "program_annotation": func_data.get("program_annotation", {}),
        "initial_execution_time": func_data.get("initial_execution_time", None)
    }
    
    # Extract schedules
    schedules = func_data.get("schedules_list", [])
    
    # Combine program details with schedules
    output_data = {
        "program_details": program_details,
        "schedules": schedules
    }
    
    # Create subfolder for the function inside output_dir
    func_subfolder = os.path.join(output_dir, function_id)
    os.makedirs(func_subfolder, exist_ok=True)
    
    # Define output file path
    output_file = os.path.join(func_subfolder, f"{function_id}_details.json")
    
    # Write to JSON file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"Created file: {output_file}")

def process_directory(input_dir, output_dir):
    """Process all JSON files in the input directory and create subfolders in the output directory."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_file = os.path.join(input_dir, filename)
            try:
                with open(input_file, 'r') as f:
                    data = json.load(f)
                
                # Each file may contain multiple functions, iterate through them
                for function_id in data.keys():
                    if function_id.startswith("function"):  # Ensure it's a function ID
                        process_function_data(data, function_id, output_dir)
            
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {input_file}: {str(e)}")
            except Exception as e:
                print(f"Error processing {input_file}: {str(e)}")

def main():
    input_dir = "Tiramisu"  # Input folder containing function JSON files
    output_dir = "converted"  # Output folder for subfolders with processed files
    
    print(f"Processing files from {input_dir} into {output_dir}")
    process_directory(input_dir, output_dir)
    print("Processing complete!")

if __name__ == "__main__":
    main()
