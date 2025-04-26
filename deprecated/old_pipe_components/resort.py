import os
import shutil

# _______________________________________________________
# GET ALL DIRECTORY FILES THAT CONTAIN DATA LIKE CSVs, etc
# _______________________________________________________


# Current working directory
cwd = os.getcwd()

# Get all folders (except specific ones)
folders = [folder for folder in os.listdir(cwd) if os.path.isdir(f'{cwd}/{folder}')]
if '__pycache__' in folders:
    folders.remove('__pycache__')

# In the original Repository these FOlders should be available
# folders.remove('original_files')
# folders.remove('.git')
# folders.remove('rewritten_files_and_tests')
print(folders)

# Collect file mappings
file_mappings = {}
for folder in folders:
    for file in os.listdir(f'{cwd}/{folder}'):
        if file == 'init.txt':
            print("init.txt shouldn't be moved")
            continue
        file_ending = file.split('.')[-1]
        file_path = f'{cwd}/{folder}/{file}'
        
        # Add to the dictionary
        if file_ending not in file_mappings:
            file_mappings[file_ending] = []  # Initialize a list for the key
        file_mappings[file_ending].append(file_path)  # Add the file path to the list

# Use similar statements to see which files have ceratin endings
# print(file_mappings['20'],'\n', file_mappings['dqs'])


# _______________________________________________________
# AFTER THE FILES HAVE BEEN DETECTED MOVE THEM TO NEW DIRS
# _______________________________________________________


# Create directories and move files
for file_ending, file_paths in file_mappings.items():
    target_dir = f'{cwd}/{file_ending}'  # Directory name based on the key
    
    # Create the directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Move each file to the corresponding directory
    for file_path in file_paths:
        target_path = f'{target_dir}/{os.path.basename(file_path)}'
        shutil.move(file_path, target_path)  # Move the file

# Debugging output
print("Files have been organized into directories:")
for key, value in file_mappings.items():
    print(f"Extension: {key}, Directory: {cwd}/{key}")
