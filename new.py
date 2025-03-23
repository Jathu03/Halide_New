import json
import os

# Sample JSON data (replace this with your actual JSON input)
json_data = '''<your JSON document here>'''  # Paste your JSON string here

# Parse the JSON data
data = json.loads(json_data)

# Base output directory
output_base_dir = "output_schedules"

# Ensure the base directory exists
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

# Iterate over each program (in this case, only "function003306")
for program_name, program_data in data.items():
    # Create a subfolder for the program
    program_dir = os.path.join(output_base_dir, program_name)
    if not os.path.exists(program_dir):
        os.makedirs(program_dir)

    # Common features to include in every file
    common_features = {
        "filename": program_data["filename"],
        "node_name": program_data["node_name"],
        "parameters": program_data["parameters"],
        "program_annotation": program_data["program_annotation"],
        "initial_execution_time": program_data["initial_execution_time"]
    }

    # Iterate over each schedule in schedules_list
    for idx, schedule in enumerate(program_data["schedules_list"]):
        # Schedule-specific data
        schedule_data = {
            "schedule_index": idx,
            "schedule_details": schedule
        }

        # Combine common features with schedule-specific data
        output_data = {**common_features, **schedule_data}

        # Define the output file name (e.g., schedule_0.json, schedule_1.json)
        output_filename = f"schedule_{idx}.json"
        output_path = os.path.join(program_dir, output_filename)

        # Write the data to a JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)

        print(f"Created file: {output_path}")

print(f"All schedules have been written to the '{output_base_dir}' directory.")
