import json
import os

# Input folder containing the program JSON files
input_folder = "Tiramisu"

# Output base directory
output_base_dir = "separate"

# Ensure the input folder exists
if not os.path.exists(input_folder):
    print(f"Error: Input folder '{input_folder}' does not exist.")
    exit(1)

# Ensure the output base directory exists
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

# Function to process a single JSON file
def process_program_file(file_path, program_name):
    # Read the JSON file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = f.read()
        data = json.loads(json_data)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{file_path}' - {e}")
        return
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    # Assuming each file has a single program as the top-level key
    for prog_key, program_data in data.items():
        # Use the program key (e.g., "function003306") or file name as the subfolder name
        subfolder_name = prog_key if prog_key else program_name.replace('.json', '')
        program_dir = os.path.join(output_base_dir, subfolder_name)

        # Create a subfolder for the program
        if not os.path.exists(program_dir):
            os.makedirs(program_dir)

        # Common features to include in every schedule file
        common_features = {
            "filename": program_data.get("filename", file_path),
            "node_name": program_data.get("node_name", "unknown"),
            "parameters": program_data.get("parameters", {}),
            "program_annotation": program_data.get("program_annotation", {}),
            "initial_execution_time": program_data.get("initial_execution_time", 0.0)
        }

        # Check if schedules_list exists
        if "schedules_list" not in program_data:
            print(f"Warning: No 'schedules_list' found in '{file_path}' for program '{prog_key}'.")
            return

        # Iterate over each schedule in schedules_list
        for idx, schedule in enumerate(program_data["schedules_list"]):
            # Schedule-specific data
            schedule_data = {
                "schedule_index": idx,
                "schedule_details": schedule
            }

            # Combine common features with schedule-specific data
            output_data = {**common_features, **schedule_data}

            # Define the output file name (e.g., schedule_0.json)
            output_filename = f"schedule_{idx}.json"
            output_path = os.path.join(program_dir, output_filename)

            # Write the data to a JSON file
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=4)
                print(f"Created file: {output_path}")
            except Exception as e:
                print(f"Error writing file '{output_path}': {e}")

# Iterate over all JSON files in the Tiramisu folder
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(input_folder, filename)
        process_program_file(file_path, filename)

print(f"All schedules have been written to the '{output_base_dir}' directory.")
