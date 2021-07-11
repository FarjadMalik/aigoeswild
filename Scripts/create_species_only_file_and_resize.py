"""

"""

import argparse
import os

import cv2
import pandas as pd


def main(input_dir, output_dir):
    # Reading input csv files
    print('Reading csv files...')
    df_images = pd.read_csv(input_dir + 'SnapshotSerengeti_v2_1_images.csv')
    df_full_annotations = pd.read_csv(input_dir + 'SnapshotSerengeti_v2_1_annotations.csv')
    # Manipulate input dataframes. Create season column and only keep s9.
    df_images.rename(columns={'Unnamed: 0': 'seq_id'}, inplace=True)
    df_images.index = df_images.capture_id
    df_images['season'] = df_images.capture_id.map(lambda x: x.split('#')[0])
    df_images['image_name'] = df_images.image_path_rel.str.split('/').str[-1]
    df_images = df_images[df_images.season.isin(['SER_S9'])]
    df_full_annotations.rename(columns={'Unnamed: 0': 'seq_id'}, inplace=True)
    df_full_annotations = df_full_annotations[df_full_annotations.capture_id.isin(df_images.capture_id)]
    print(f"Merging the images and the annotations...")
    # Join the images and their annotations. Keep only interesting columns
    df_full = df_images.join(df_full_annotations.set_index('capture_id'), how='inner', on=None, lsuffix='_x',
                             rsuffix='_y')
    df_full = df_full[['capture_id', 'image_rank_in_capture', 'image_path_rel', 'image_name',
                       'season_x', 'question__species', 'site']]

    # Removing these not needed anymore
    del df_images
    del df_full_annotations

    # Remove images with no species
    df_full = df_full.loc[~df_full.question__species.str.contains('blank')]
    # Store this as a csv file
    species_df_filepath = os.path.join(output_dir, "ser_s9_species_df.csv")
    df_full.to_csv(species_df_filepath, sep=',', index=False)

    # Remove it later on. Only needed because some images were not resized
    # print(f"Left join the images already processed...")
    # # Read the images which have already been resized and remove them from left join
    # df_downloaded_images = pd.read_csv(os.path.join(input_dir, r'backup\train_species_only.csv'))
    # df_downloaded_images['local_file_name'] = df_downloaded_images.file_path_rel.str.split('/').str[-1]
    # df_missing_left = pd.merge(df_full, df_downloaded_images, how='left',
    #                            left_on="image_name",
    #                            right_on="local_file_name")
    # Merge it and then remove it
    # df_downloaded_images
    # # Keep only the interested columns
    # df_missing_left = df_missing_left[
    #     ['capture_id', 'image_path_rel', 'question__species', 'file_path_rel', 'encoded_species']]
    # # Remove images that have already been processed
    # df_missing = df_missing_left[df_missing_left.isnull().any(axis=1)]
    # del df_missing_left

    print(f"Creating filename local...")
    # Create the column with the full image paths
    base_image_dir = os.path.join(input_dir, r'snapshotserengeti-unzipped\snapshotserengeti-unzipped')
    df_full['file_path_local'] = df_full.image_path_rel.map(lambda x: os.path.join(base_image_dir, x))

    print(f"Resizing...")
    # resize the images and return the the lists
    resize_images(df_full.file_path_local)


def resize_images(filenames):
    exists = []
    # Loop over the files and resize and write it back on the same path using cv2
    for index, file_name in enumerate(filenames):
        if os.path.isfile(file_name):
            image = cv2.imread(file_name)
            if image is not None:
                image = cv2.resize(image, (256, 256))
                cv2.imwrite(file_name, image)

            else:
                print(f"Found the image but cannot open with cv2: {file_name}")
        else:
            print(f"Cannot open file. Doesnt exist: {file_name}")
        if index % 1000 == 0:
            print(index)
        if index % 100000 == 0:
            print(index)

    print(f"---resize_images---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create_species_only_file_and_resize.py')
    # Add parser arguments
    parser.add_argument("--input-directory", default=r"C:\Users\mfarj\Documents\ss_data\\", type=str,
                        help="Path to the input image csv")
    parser.add_argument("--output-directory", default=r"C:\Users\mfarj\Documents\ss_data\data_csv\\", type=str,
                        help="Path to the output csv")
    # Execute the parse_args() method
    args = parser.parse_args()

    main(args.input_directory, args.output_directory)
