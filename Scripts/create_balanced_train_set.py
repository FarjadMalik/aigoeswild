import argparse
import os

import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def sample_class(class_set, desired_count, seed=42):
    # Create empty dataframes
    balanced_set = pd.DataFrame(columns=class_set.columns)

    class_set = shuffle(class_set, random_state=seed)
    balanced_set = class_set.sample(n=desired_count, replace=len(balanced_set) < desired_count,
                                    random_state=seed, axis=0)

    # check if the count was reached
    assert (len(balanced_set) == desired_count)

    return balanced_set


def sample_dataset(dataset=None, dist=None):
    if dataset is None or dist is None:
        print(f"Input dataset and/or distribution is missing.\n Dataset:{dataset} Dist: {dist}")
        return None
    print(len(dist))
    print(len(dataset['encoded_species'].unique()))
    # check if the len of distinct species and our distribution is the same
    # assert (len(dist) == len(dataset['encoded_species'].unique()))

    # Create empty dataframes
    output_dataset = pd.DataFrame(columns=dataset.columns)

    # For each specie divide into train and test sets
    for index, row in dist.iterrows():
        # For testing
        # specie = row['Species']
        # if not specie == 'reptiles':
        #     continue
        class_label = row['class_label']
        count_desired = row['desired_count']
        if np.isnan(class_label):
            print(f"Sampling dataset for label: {class_label} is not possible. continuing..")
            continue

        print(f"Sampling dataset for label: {class_label}")
        class_subset = dataset.loc[dataset['encoded_species'] == class_label]
        class_count = len(class_subset)
        print(f"Number of images for class label {class_label} is: {class_count}")
        print(f"Desired Number of images for class label {class_label} is: {count_desired}")

        # Randomly oversample or under sample the given class to the desired count
        class_balanced = sample_class(class_set=class_subset, desired_count=int(count_desired))

        # Add the balanced class dataset to the output dataset
        output_dataset = output_dataset.append(class_balanced)
        output_dataset = shuffle(output_dataset, random_state=42)

    return output_dataset


def main(file_train, file_distribution, column, output_dir):
    # Read the original dataset
    og_train_df = pd.read_csv(file_train)
    print(f"Len of original train dataset: {len(og_train_df)}")
    print(f"Train dataset Columns: \n{og_train_df.columns}")

    # Reading the input excel file containing data distribution and preping it for use
    data_dist = pd.read_excel(file_distribution, engine='openpyxl')
    # Remove space from column names
    data_dist.columns = data_dist.columns.str.replace(' ', '')

    print(f"Data distribution file columns: \n{data_dist.columns}")
    # Data dist file has many columns for different models so here we specify and extract the
    # column to be used for the output dataset
    if column is None or column > len(data_dist.columns):
        print(f"Either no column specified or the column is outside the range: {column}")
        return
    column_name = data_dist.columns[column - 1]
    print(f"Column to use is # {column} named: {column_name}")
    data_dist = data_dist.iloc[:, [0, 1, column - 1]]
    # should now contain only 3 columns, one of specie names, one for their label and one for the desired count
    assert(len(data_dist.columns) == 3)
    # Update column names and then show data distribution columns
    data_dist.columns = ['species', 'class_label', 'desired_count']
    print(f"Data distribution file columns after prep: \n{data_dist.columns}")

    """
    # Once last check with the specie mapping file to confirm if this is indeed the correct labelling
    # 
    """
    # file_label_mapping = r"C:\Users\mfarj\Documents\ss_data\data_csv\label_to_species_mapping.csv"
    # specie_encoding = pd.read_csv(file_label_mapping)
    # print(f"Len of specie_encoding: {len(specie_encoding)}")
    # print(f"specie_encoding columns: \n{specie_encoding.columns}")
    # print(f"specie_encoding head: \n{specie_encoding.head}")
    #
    # joined_df = pd.merge(left=data_dist, right=specie_encoding, how='inner', right_on='encoded_species',
    #                      left_on='class label ')
    # print(f"Len of joined_df: {len(joined_df)}")
    # print(f"joined_df columns: \n{joined_df.columns}")
    # print(f"joined_df head: \n{joined_df.head}")
    # print(f"Joined df where species not equal: {joined_df.loc[joined_df['Species ']
    #         != joined_df['question__species']]}")
    # assert(joined_df.loc[joined_df['Species '] != joined_df['question__species']] is not None)
    """
    # 
    # check okay
    """

    # We will now divide the dataset based on the two files
    balanced_train = sample_dataset(dataset=og_train_df, dist=data_dist)

    # Store as csv
    fn_balanced_input = os.path.join(output_dir, f"{column_name}_train_input.csv")
    fn_balanced_set = os.path.join(output_dir, f"{column_name}_train_set.csv")
    print(f"Number of images in balanced train dataset: {len(balanced_train)}")
    balanced_train.to_csv(fn_balanced_input, columns=['image_path_rel', 'encoded_species'], sep=',', index=False)
    balanced_train.to_csv(fn_balanced_set, sep=',', index=False)

    # Group them by species and sort in ascending orders
    count_per_specie = balanced_train.groupby(['question__species']).capture_id.count().reset_index(
        name='count').sort_values(['count'], ascending=False)
    fn_balanced_cps = f"{column_name}_train_count_per_species.csv"
    count_per_specie.to_csv(os.path.join(output_dir, fn_balanced_cps), sep=',', header=True, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create_species_only_file_and_resize.py')
    # Add parser arguments
    parser.add_argument("--input-train", default=r"C:\Users\mfarj\Documents\ss_data\data_csv\train_set.csv",
                        type=str,
                        help="Path to the original train set")
    parser.add_argument("--data-distribution", default=r"C:\Users\mfarj\Documents\ss_data\data_csv\Data_distributions"
                                                       r".xlsx",
                        type=str,
                        help="Path to the desired data distribution file")
    parser.add_argument("--column", default=9,
                        type=int,
                        help="Column (c) to choose from the excel dist (used as c-1)")
    parser.add_argument("--output-directory", default=r"C:\Users\mfarj\Documents\ss_data\data_csv", type=str,
                        help="Path for the output set")
    # Execute the parse_args() method
    args = parser.parse_args()

    main(args.input_train, args.data_distribution, args.column, args.output_directory)
