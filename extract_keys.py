import os
import json
import re

# Define the keys to search for
keys_to_search = ["YS", "UTS", "EL", "RBAR", "NVALM", "Ra"]

# Define the replacements for substrings
substring_replacements = { 
    "YS MPA": "YS",
    "UTS MPA": "UTS",
    "EL %": "EL",
    "E\nRVALU": "RBAR",
    "NVALM": "NVALM",
    "Ra MICRO M": "Ra"
}

# Directory containing the JSON files
input_directory = 'jcpal_output_json/'  # Update this path to your directory

# Directory to save the processed results
output_directory = os.path.join(input_directory, 'final_output')
os.makedirs(output_directory, exist_ok=True)  # Create the output directory if it doesn't exist

# Function to process each file
def process_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        
        # Preprocessing: Replace substrings in keys
        new_data = {}
        for key in data.keys():
            new_key = key
            for pattern, replacement in substring_replacements.items():
                if re.match(pattern, key):
                    new_key = replacement
                    break
            new_data[new_key] = data[key]
        data = new_data

        # Initialize the result dictionary
        result = {}
        
        # Process each key
        for key in keys_to_search:
            if key in data:
                values = data[key]
                if len(values) > 2:
                    result[substring_replacements.get(key, key)] = values[len(values)-1]  # Use the replaced key if it exists
                else:
                    result[substring_replacements.get(key, key)] = "Data Unavailable"
            else:
                result[substring_replacements.get(key, key)] = "Data Unavailable"
        
        return result

# Process all files in the directory
def process_all_files(directory):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            result = process_file(file_path)
            results.append({filename: result})
    
    # Save the results to a single JSON file
    output_file_path = os.path.join(output_directory, 'processed_results.json')
    with open(output_file_path, 'w') as output_file:
        json.dump(results, output_file, indent=4)

# Run the processing function
process_all_files(input_directory)