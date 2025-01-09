import os
import re
from Ultrasound.Dataset.UltrasoundDataset import cadaver_ids

def extract_timestamps(directory):
    """
    Extract timestamps from filenames in a given directory that match the pattern "timestamp_label_overlap.png".

    Parameters:
    - directory (str): The path to the directory containing the files.

    Returns:
    - list: A list of timestamps extracted from the filenames.
    """
    # Regular expression to match filenames and capture the timestamp
    pattern = r'(\d+)_label_overlap\.png'

    # List to store the timestamps
    timestamps = []

    # Loop over the files in the directory
    for filename in os.listdir(directory):
        # Check if the filename matches the pattern
        match = re.match(pattern, filename)
        if match:
            # Extract the timestamp (first group of the match)
            timestamp = int(match.group(1))
            timestamps.append(timestamp)
    timestamps.sort()

    return timestamps

def get_file_paths_in_directory(directory,full_path=True):
    """
    Generate a list of paths to all files directly contained within a directory,
    without descending into subdirectories.

    :param directory: The path to the directory from which file paths are to be retrieved.
    :return: A list of file paths.
    """
    file_paths = []
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isfile(full_path):
            if full_path:
                file_paths.append(full_path)
            else:
                file_paths.append(item)
    return file_paths
# Usage example (Replace the directory path with your actual folder path)

def rename_folders(root_folder, original_name, new_name):
    """
    Renames all folders with the name 'original_name' to 'new_name' within the 'root_folder'.

    :param root_folder: The root directory to start searching from.
    :param original_name: The original name of the folders to be renamed.
    :param new_name: The new name to apply to the folders.
    """
    # Walk through all directories and subdirectories starting from root_folder
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # Iterate through each subdirectory in the current directory
        for dirname in dirnames:
            # Check if the current directory name matches the original_name
            if dirname == original_name:
                # Construct full path to the current directory
                full_dir_path = os.path.join(dirpath, dirname)
                # Construct the new path with the new name
                new_dir_path = os.path.join(dirpath, new_name)
                # Rename the directory
                os.rename(full_dir_path, new_dir_path)
                print(f"Renamed '{full_dir_path}' to '{new_dir_path}'")

def delete_files_in_folder(directory):
    # List all files and directories in the given directory
    for item in os.listdir(directory):
        # Create the full path to the item
        item_path = os.path.join(directory, item)
        # Check if the item is a file
        if os.path.isfile(item_path):
            # Delete the file
            os.remove(item_path)

def main_rename_folders():
    root_folder = "Z:/AI_Ultrasound_dataset"
    original_name = "ImageLabelOverlap_SDF_new"
    new_name = "ImageLabelOverlap_full"
    rename_folders(root_folder, original_name, new_name)

    original_name = "Label_SDF_new"
    new_name = "Label_full"
    rename_folders(root_folder, original_name, new_name)

def main_clean_redudent_labels():
    dataset_root_folder = "Z:/AI_Ultrasound_dataset"
    cadavers_involved = list(range(1, 15))  # Adjust the range as needed
    dataFolders = []
    for idx in cadavers_involved:
        cadaver_id = cadaver_ids[idx]  # Update according to how cadaver_ids are formatted
        dataFolders += [f"{dataset_root_folder}/{cadaver_id}/Linear18/record{i:02d}" for i in range(1, 15)]
    for dataFolder in dataFolders:
        for label_folder in [os.path.join(dataFolder,"Label_full"),os.path.join(dataFolder,"Label_partial_gradient")]:
            i=0
            if not os.path.isdir(dataFolder):
                continue
            label_overlap_folder=os.path.join(dataFolder,"ImageLabelOverlap_partial_gradient")
            timestamps_target = [int(file[:file.find('_')]) for file in os.listdir(label_overlap_folder) if
                                 os.path.isfile(os.path.join(label_overlap_folder, file))]
            timestamps_target = set(timestamps_target)
            for file in os.listdir(label_folder):
                timestamp=int(file[:file.find('_')])
                if not timestamp in timestamps_target:
                    os.remove(os.path.join(label_folder,file))
                    i+=1
            print(f"remove {i} files in {dataFolder}")





if __name__=="__main__":
    main_clean_redudent_labels()