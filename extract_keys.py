import os
import json
import re

# Define the keys to search for
keys_to_search = ["YS", "UTS", "EL", "RBAR", "NVALM", "Ra"]

# Define the replacements for substrings
substring_replacements = {
    "YS (Mpa)": "YS",
    "(MPa)\nUTS": "UTS",
    "EL%": "EL",
    "R_90": "RBAR",
    "N_90": "NVALM",
    "Raum": "Ra"
}

# Directory containing the JSON files
input_directory = '/home/hi-born4/Bristlecone/AI-powered document processing and extraction system/jsw_output_json'  # Update this path to your directory

# Directory to save the processed results
output_directory = os.path.join(input_directory, 'final_output')

# Function to process each file
def process_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        
        # Preprocessing: Replace substrings in keys
        for key in list(data.keys()):
            for substring in substring_replacements:
                pattern = re.compile(substring.replace("%", "."))
                if pattern.match(key):
                    new_key = key.replace(substring, substring_replacements[substring])
                    data[new_key] = data.pop(key)
        
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
            result = process_file(file_path)
            results.append(result)
    
    return results

# Save the processed results to a new JSON file
def save_results(results, output_file):
    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

# Run the processing and save the results
def main(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    results = process_all_files(input_directory)
    output_file = os.path.join(output_directory, 'processed_results.json')
    save_results(results, output_file)
    print(f"Processed results saved to {output_file}")

# Execute the main function
main(input_directory, output_directory)
