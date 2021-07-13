"""

"""
import argparse
import pandas as pd
import os


def main(input_dir, output_dir):
    # Create the input file paths names
    train_path = os.path.join(input_dir, 'train_set.csv')
    test_path = os.path.join(input_dir, 'test_set.csv')
    valid_path = os.path.join(input_dir, 'valid_set.csv')

    # Read the csv files
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    valid_df = pd.read_csv(valid_path)

    # Group them by species and sort in ascending orders
    train_count_per_specie = train_df.groupby(['question__species']).capture_id.count().reset_index(
        name='count').sort_values(['count'], ascending=False)
    test_count_per_specie = test_df.groupby(['question__species']).capture_id.count().reset_index(
        name='count').sort_values(['count'], ascending=False)
    valid_count_per_specie = valid_df.groupby(['question__species']).capture_id.count().reset_index(
        name='count').sort_values(['count'], ascending=False)

    # Store them as a csv
    train_count_per_specie.to_csv(os.path.join(output_dir, 'train_count_per_species.csv'), sep=',', header=True, index=False)
    test_count_per_specie.to_csv(os.path.join(output_dir, 'test_count_per_species.csv'), sep=',', header=True, index=False)
    valid_count_per_specie.to_csv(os.path.join(output_dir, 'valid_count_per_species.csv'), sep=',', header=True, index=False)

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create_species_only_file_and_resize.py')
    # Add parser arguments
    parser.add_argument("--input-directory", default=r"C:\Users\mfarj\Documents\ss_data\data_csv",
                        type=str,
                        help="Path to the input image csv containing species")
    parser.add_argument("--output-directory", default=r"C:\Users\mfarj\Documents\ss_data\data_csv", type=str,
                        help="Path to the output csv")
    # Execute the parse_args() method
    args = parser.parse_args()

    main(args.input_directory, args.output_directory)


