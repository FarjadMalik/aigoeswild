"""

"""
import pandas as pd

# Path prefix to be used for the image_names in the csvs
path_prefix = r'E:\ss_data\snapshotserengeti-unzipped\snapshotserengeti-unzipped'
# Read the latest train csv
train_csv = r'E:\ss_data\train_phase2_v6_fixed.csv'
train_df = pd.read_csv(train_csv)
print(f"Number of csv files v1: {len(train_df)}")
# Append the other csv files as well

train_csv_v2 = r'E:\ss_data\train_phase2_v7_fixed.csv'
train_df_v2 = pd.read_csv(train_csv_v2)
train_df_v2.columns = ['file_path', 'species']
print(f"Number of csv files v2: {len(train_df_v2)}")
print(f'Train df v2: \n{train_df_v2.head()}')
print(f"Total number of samples: {len(train_df_v2)+len(train_df)}")

train_df = pd.concat([train_df, train_df_v2], axis=0, ignore_index=True)
print(f"Number of csv files: {len(train_df)}")

# Take a look at the unique species in our datasets
print(f"Unique species labels in our dataset: {train_df.species.unique()}")

# Read csv for labels enumeration
species_enumeration = r'E:\ss_data\label_to_species_V3.csv'
enum_species_df = pd.read_csv(species_enumeration)
print(f"Number of species: {len(enum_species_df)}")
print(f'{enum_species_df.head()}')

# Join species with the train csv to get the correct labels
# Identify joining columns
print(f"Train df columns: {train_df.columns}")
print(f"Species enum columns: {enum_species_df.columns}")
# join on species and encoded_species
train_df = pd.merge(left=train_df, right=enum_species_df, how='inner', left_on='species',
                    right_on='encoded_species')
train_df.drop('species_x', axis=1, inplace=True)
train_df.columns = ['file_path_rel', 'species', 'encoded_species']
print(f'Merged train df: \n{train_df.head()}')

# Filter on the dataset with species only
species_train_df = train_df.loc[(train_df.encoded_species > 0)]
print(f'Species only train df: \n{species_train_df.head()}')
print(f'Species only size: \n{len(species_train_df)}')
# Store this as a csv
species_train_path = r'E:\ss_data\train_species_only.csv'
species_train_df.to_csv(species_train_path, sep=',', header=True, index=False, columns=['file_path_rel',
                                                                                        'encoded_species'])

# Extract a df that contains only 1 species
duplicated_species_df = species_train_df[species_train_df.duplicated(['file_path_rel'])]
print(f'Duplicated files : \n{duplicated_species_df.head()}')
print(f'Duplicated files length: \n{len(duplicated_species_df)}')
print(f'Unique files: \n{duplicated_species_df.file_path_rel.nunique()}')
# We see that this list still contains double capture_id's meaning that
# some images (7128-7065= 63) contain more than 2 species
# cond1 = files_with_multiple_species Remove them from our train csv
cond1 = species_train_df['file_path_rel'].isin(duplicated_species_df['file_path_rel'])
# for the rows where condition = true -> row is dropped cause it refers to image containing multiple species
single_species_df = species_train_df.drop(species_train_df[cond1].index)
print(f'Train csv df with only one species: \n{single_species_df.head()}')
print(f'Number of files with one species: \n{len(single_species_df)}')
# Store this as a csv
single_species_train_path = r'E:\ss_data\train_species_only_singles.csv'
single_species_df.to_csv(single_species_train_path, sep=',', header=True, index=False, columns=['file_path_rel',
                                                                                                'encoded_species'])
