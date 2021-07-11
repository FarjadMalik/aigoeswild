"""
Script to move the images with file path relative to a new location keeping their structure intact
"""
import os
from shutil import copyfile

import pandas as pd

# Input image dirs
input_csv = r'E:\ss_data\train_species_only.csv'
base_dir = r'E:\ss_data\snapshotserengeti-unzipped\snapshotserengeti-unzipped'
output_dir = r'C:\Users\mfarj\Documents\ss_data\snapshotserengeti-unzipped\snapshotserengeti-unzipped'

# read input csv
images_df = pd.read_csv(input_csv)
# Create the full path and the path we want to move it to
images_df['file_path_full'] = images_df.file_path_rel.map(lambda x: os.path.join(base_dir, x))
images_df['output_path'] = images_df.file_path_rel.map(lambda x: os.path.join(output_dir, x))
print(f"File Count: {len(images_df)}")

# Iterate over the dataframe
for index, row in images_df.iterrows():
    # if index == 1:
    #     break
    # print(f"Index: {index}")
    # print(f"file path rel: {row['file_path_rel']}")
    # print(f"file path full: {row['file_path_full']}")
    # print(f"file path output: {row['output_path']}")

    #
    if index % 10000 == 0:
        print(index)
    # Check if the path to the file exists. If it doesnt create it using os make dir
    os.makedirs(os.path.dirname(row['output_path']), exist_ok=True)
    # Once path is fully there we can move the file using shutil copy file
    copyfile(row['file_path_full'], row['output_path'])
# End
print(f'Finished moving the files to {output_dir}')
