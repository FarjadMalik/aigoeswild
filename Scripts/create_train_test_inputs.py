import argparse
import os

import pandas as pd
from sklearn.utils import shuffle


def split_species(specie_set, train_count, test_count, seed=42):
    # Create empty dataframes
    train_set = pd.DataFrame(columns=specie_set.columns)
    test_set = pd.DataFrame(columns=specie_set.columns)

    # Get the counts for each capture event
    capture_ids = specie_set.groupby(['capture_id']).image_rank_in_capture.count().reset_index(
        name='count')
    # shuffle them randomly
    capture_ids = shuffle(capture_ids, random_state=seed)
    for index, row in capture_ids.iterrows():
        # get images for each capture id
        capture_subset = specie_set.loc[specie_set['capture_id'] == row['capture_id']]
        capture_count = len(capture_subset)
        count_left = test_count - len(test_set)

        # if count is less than test add them to the test
        if capture_count > count_left:
            continue
        elif capture_count <= count_left:
            test_set = test_set.append(capture_subset)

        # break condition when all test images have been extracted
        if len(test_set) >= test_count:
            train_set = specie_set.loc[~specie_set.index.isin(test_set.index)]
            break

    print(f"Train count for {len(train_set)} and the rest {len(test_set)} will be used for testing.")
    return train_set, test_set


def split_dataset(data=None, test_ratio=0.2, seed=42):
    if data is None:
        print(f"No input data provided")
        return None, None

    # Create empty dataframes
    train = pd.DataFrame(columns=data.columns)
    test = pd.DataFrame(columns=data.columns)

    # Get distinct species
    distinct_species = data.question__species.unique()
    print(f"Number of distinct species: {len(distinct_species)}")

    # For each specie divide into train and test sets
    for specie in distinct_species:
        print(f"Splitting dataset for {specie}")
        specie_subset = data.loc[data['question__species'] == specie]
        specie_count = len(specie_subset)
        test_count = int(specie_count * test_ratio)
        train_count = specie_count - test_count
        train_species, test_species = split_species(specie_set=specie_subset, train_count=train_count,
                                                    test_count=test_count, seed=seed)
        train = train.append(train_species)
        test = test.append(test_species)

    # Shuffle them a bit so that we dont get all species in the same batches
    train = shuffle(train)
    test = shuffle(test)

    return train, test


def main(path_csv, output_dir):
    base_image_dir = r'C:\Users\mfarj\Documents\ss_data\snapshotserengeti-unzipped\snapshotserengeti-unzipped'
    # Read input csv
    species_df = pd.read_csv(path_csv)
    # add column for full path
    species_df['file_path_local'] = species_df.image_path_rel.map(lambda x: os.path.join(base_image_dir, x))

    print(f"Len of images with one species in season 9: {len(species_df)}")
    print(f"Species df Columns: \n{species_df.columns}")

    # logic to divide first based on images with multiple species. keep only images with one specie
    # Images with multiple species in it
    # Extract a df that contains only 1 species
    # duplicated_species_df = species_df[species_df.duplicated(['image_path_rel'])]
    # print(f'Duplicated files : \n{duplicated_species_df.head()}')
    # print(f'Duplicated files length: \n{len(duplicated_species_df)}')
    # # We see that this list still contains double capture_id's meaning that some images (7128-7065= 63)
    # # contain more than 2 species
    # print(f'Unique files: \n{duplicated_species_df.image_path_rel.nunique()}')
    # # cond1 = files_with_multiple_species Remove them from our train csv
    # cond1 = species_df['image_path_rel'].isin(duplicated_species_df['image_path_rel'])
    # # for the rows where condition = true -> row is dropped cause it refers to image containing multiple species
    # single_species_df = species_df.drop(species_df[cond1].index)
    # print(f'Train csv df with only one species: \n{single_species_df.head()}')
    # print(f'Number of files with one species: \n{len(single_species_df)}')
    # # Store this as a csv
    # single_species_train_path = os.path.join(output_dir, 's9_with_single_species.csv')
    # single_species_df.to_csv(single_species_train_path, sep=',', header=True, index=False)

    # See the counts of duplicated capture ids
    same_capture_df = species_df[species_df.duplicated(['capture_id'])].capture_id
    print(f'Count of rows with same capture id: \n{len(same_capture_df)}')

    # Then based on capture id. Make sure all same capture ids are in the same sets (train/valid/test)
    train_df, test_df = split_dataset(data=species_df, test_ratio=0.2, seed=42)

    # Store as csv
    train_file_name = os.path.join(output_dir, "train_phase2_split.csv")
    train_full_df_path = os.path.join(output_dir, "train_set.csv")
    print(f"Number of images in train dataset: {len(train_df)}")
    train_df.to_csv(train_file_name, columns=['image_path_rel', 'encoded_species'], sep=',', index=False)
    train_df.to_csv(train_full_df_path, sep=',', index=False)

    test_file_name = os.path.join(output_dir, "test_phase2_split.csv")
    test_full_df_path = os.path.join(output_dir, "test_set.csv")
    print(f"Number of images in test dataset: {len(test_df)}")
    test_df.to_csv(test_file_name, columns=['image_path_rel', 'encoded_species'], sep=',', index=False)
    test_df.to_csv(test_full_df_path, sep=',', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create_species_only_file_and_resize.py')
    # Add parser arguments
    parser.add_argument("--input-csv", default=r"C:\Users\mfarj\Documents\ss_data\data_csv\s9_with_single_species"
                                               r".csv",
                        type=str,
                        help="Path to the input image csv")
    parser.add_argument("--output-directory", default=r"C:\Users\mfarj\Documents\ss_data\data_csv", type=str,
                        help="Path to the output csv")
    # Execute the parse_args() method
    args = parser.parse_args()

    main(args.input_csv, args.output_directory)
