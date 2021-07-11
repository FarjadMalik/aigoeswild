"""

"""
import os

import cv2
import numpy as np
import pandas as pd

if __name__ == "__main__":
    path_csv = r'E:\ss_data\train_phase2_v7.csv'
    train_df = pd.read_csv(path_csv)
    print(len(train_df))
    print(train_df.encoded_species.nunique())
    print(train_df.head())
    base_dir = r'E:\ss_data\snapshotserengeti-unzipped\snapshotserengeti-unzipped'
    train_df['file_path_local'] = train_df.rel_path.map(lambda x: os.path.join(base_dir, x))
    print(train_df.head())

    rel_path = []
    species_found = []
    encoded_species = []
    start_from = 0
    end_at = 1000000
    for index, file_name in enumerate(train_df.file_path_local[start_from:]):
        #     print(file_name)
        #     print(index)
        #     print(train_df.file_path[index])
        #     print(train_df.file_path_local[index])
        #     print(train_df.encoded_species[index])

        index = start_from + index
        if index == (start_from + end_at):
            print(f'Breaking at size {index}')
            break

        image = cv2.imread(file_name)
        if image is not None:
            #         print(image.shape)
            image = cv2.resize(image, (256, 256))
            #         print(image.shape)
            cv2.imwrite(file_name, image)
            rel_path.append(train_df.file_path[index])
            species_found.append(0 if train_df.encoded_species[index] == 0 else 1)
            encoded_species.append(train_df.encoded_species[index])
        else:
            if os.path.isfile(file_name):
                print(f'Removing {file_name}')
                try:
                    os.remove(os.path.join(file_name))
                except NotImplementedError as e:
                    continue
            continue
        if index % 10000 == 0:
            print(index)
        if index % 100000 == 0:
            print(index)

    print(
        f'Lengths: RelativePath  = {len(rel_path)}.  species_found  = {len(species_found)}.  '
        f'encoded_species  = {len(encoded_species)}')
    train_df_filtered = pd.DataFrame(np.column_stack([rel_path, species_found, encoded_species]),
                                     columns=['rel_path', 'boolean_species', 'encoded_species'])
    print(f'Filtered Dataframe size {len(train_df_filtered)}')
    output_filepath = os.path.join(base_dir, "train_phase1_v7.csv")
    train_df_filtered.to_csv(output_filepath, columns=['rel_path', 'boolean_species'], sep=',', index=False)

    output_filepath = os.path.join(base_dir, "train_phase2_v7.csv")
    train_df_filtered.to_csv(output_filepath, columns=['rel_path', 'encoded_species'], sep=',', index=False)
