import argparse
import os

import pandas as pd


def main(path_csv, output_dir):
    base_image_dir = r'C:\Users\mfarj\Documents\ss_data\snapshotserengeti-unzipped\snapshotserengeti-unzipped'
    # Read input csv
    species_df = pd.read_csv(path_csv)
    print(f"Len of images with species in season 9: {len(species_df)}")

    # add column for full path
    species_df['file_path_local'] = species_df.image_path_rel.map(lambda x: os.path.join(base_image_dir, x))

    # file_doesnt_exist = []
    # print(f"Iterating over files to see if they exist:")
    # for index, row in species_df.iterrows():
    #     if index % 10000 == 0:
    #         print(index)
    #     if not os.path.isfile(row['file_path_local']):
    #         file_doesnt_exist.append(row.file_path_local)
    #         species_df.drop(index, inplace=True)
    #
    # print(f"Len after filtering non-existent files: {len(species_df)}")
    # print(f"Len missing: {len(file_doesnt_exist)}")
    #
    # # Store the missing images to a csv
    # missing_files_path = os.path.join(output_dir, "missing_files.csv")
    # with open(missing_files_path, 'w', newline='') as f:
    #     wr = csv.writer(f, quoting=csv.QUOTE_ALL)
    #     wr.writerow(file_doesnt_exist)
    #

    # Creating a encoded csv with relative paths and species to use for training
    df_train_phase2 = species_df[['image_path_rel', 'question__species']]
    enc_species, name_species = pd.factorize(df_train_phase2['question__species'])
    df_train_phase2['encoded_species'] = enc_species

    # Store filtered df as a csv file
    species_df['encoded_species'] = enc_species
    species_df_filepath = os.path.join(output_dir, "ser_s9_species_df_filtered.csv")
    # species_df.to_csv(species_df_filepath, sep=',', index=False)

    # Create a label to specie mapping df
    df_labels = df_train_phase2.groupby(by=['question__species', 'encoded_species'], as_index=False).first()
    # Store as csv
    label_file_name = os.path.join(output_dir, "label_to_species_mapping.csv")
    # df_labels.to_csv(label_file_name, columns=['question__species', 'encoded_species'], sep=',', index=False)
    print(f"---main---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create_species_only_file_and_resize.py')
    # Add parser arguments
    parser.add_argument("--input-csv", default=r"C:\Users\mfarj\Documents\ss_data\data_csv\ser_s9_species_df_filtered.csv",
                        type=str,
                        help="Path to the input image csv")
    parser.add_argument("--output-directory", default=r"C:\Users\mfarj\Documents\ss_data\data_csv", type=str,
                        help="Path to the output csv")
    # Execute the parse_args() method
    args = parser.parse_args()

    main(args.input_csv, args.output_directory)
