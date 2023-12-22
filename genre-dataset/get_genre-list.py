import yaml
import json
import re
from unidecode import unidecode

# Defining functions
# Load YAML file
def load_yaml(file_path): 
    with open(file_path, 'r') as file: 
        data = yaml.safe_load(file)
    return data

# Flat data to a list (ignoring the level of the elements)
def flatten_recursive(data):
    result = []

    if isinstance(data, list):
        for item in data:
            result.extend(flatten_recursive(item))
    elif isinstance(data, dict):
        for key, value in data.items():
            result.extend([f"{subkey.lower()}" for subkey in flatten_recursive(value)])
    else:
        result.append(str(data).lower())

    return result

# Clear accents & other characters from a list
def remove_accents(input_str):
    return unidecode(input_str)

def remove_symbols(input_str):
    # Using a regular expression to remove symbols
    return re.sub(r'[^\w\s]', '', input_str)

############################################################
# Path to input YAML files
list1_path = 'shared-genres_list_D400-AM-EveryNoise.yaml'
list2_path = 'wiki-popular-2023.yaml'
############################################################

# Load data from YAML files
list1 = load_yaml(list1_path)
list2 = load_yaml(list2_path)

# Flat data to a list 
flattened_list_1 = flatten_recursive(list1)
flattened_list_2 = flatten_recursive(list2)

# Remove all the wierd non unicode characters
processed_list_1 = [remove_symbols(remove_accents(item)) for item in flattened_list_1]
processed_list_2 = [remove_symbols(remove_accents(item)) for item in flattened_list_2]

# Find shared elements between both lists 
shared_elements  = [element for element in processed_list_1 if element in processed_list_2]

print("Shared elements:", len(shared_elements))

# Store the shared elements 
with open('shared-genres_list_D400-AM-EveryNoise-Wiki.yaml', 'w') as yaml_file: 
    yaml.dump({'Common Genres D400 & AM & EveryNoise & Wikipedia': shared_elements}, yaml_file,  default_flow_style=False)