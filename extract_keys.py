import os
import json
import re

# Define the keys to search for
keys_to_search = ["YS", "UTS", "EL", "RBAR", "NVALM", "Ra"]

# Define the replacements for substrings
substring_replacements = {
    ".*YS.*": "YS",
    ".*UTS.*": "UTS",
    ".*EL": "EL",
    ".*R_90.*": "RBAR",
    ".*N_90.*": "NVALM",
    ".*Ra.*": "Ra"
}

# Directory containing the JSON files
input_directory = '/home/hi-born4/Bristlecone/AI-powered document processing and extraction system/jsw_output_json/'  # Update this path to your directory

# Directory to save the processed results
output_directory = os.path.join(input_directory, 'final_output')

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
                    result[key] = values[2]  # Use the original key
                else:
                    result[key] = "Data Unavailable"
            else:
                result[key] = "Data Unavailable"
        
        return result

# Process all files in the directory
def process_all_files(directory):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
