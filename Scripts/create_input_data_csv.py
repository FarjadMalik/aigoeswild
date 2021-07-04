import os
import pandas as pd

# This is where our downloaded images and metadata live locally
csv_dir = os.path.join("c:/", "Users", "mfarj", "Documents", "ss_data")
base_dir = os.path.join("e:/", "ss_data")
src_dir = os.path.join(base_dir, "snapshotserengeti-unzipped", "snapshotserengeti-unzipped")
season = 'SER_S9'


if __name__ == '__main__':
    df_full_images = pd.read_csv(csv_dir + r'\SnapshotSerengeti_v2_1_images.csv')
    df_full_annotations = pd.read_csv(csv_dir + r'\SnapshotSerengeti_v2_1_annotations.csv')
    df_full_images.rename(columns={'Unnamed: 0': 'seq_id'}, inplace=True)
    df_full_annotations.rename(columns={'Unnamed: 0': 'seq_id'}, inplace=True)
    df_full_images.index = df_full_images.capture_id
    df_full_images['season'] = df_full_images.capture_id.map(lambda x: x.split('#')[0])
    df_full_images = df_full_images[df_full_images.season.isin([('%s' % season)])]
    df_full_annotations = df_full_annotations[df_full_annotations.capture_id.isin(df_full_images.capture_id)]

    df_full_images['file_name_local'] = df_full_images.apply(
        lambda x: (src_dir + '/' + x.image_path_rel), axis=1
    )
    df_full = df_full_images.join(df_full_annotations.set_index('capture_id'), how='inner', on=None, lsuffix='_x',
                                  rsuffix='_y')

    print('DF Full length:', len(df_full))

    file_path_rel = []
    question_species = []

    for index, row in df_full.iterrows():
        if os.path.isfile(row['file_name_local']) and os.stat(row['file_name_local']).st_size > 0:
            file_path_rel.append(row['image_path_rel'])
            question_species.append(row['question__species'])

    print('Number of images: ', len(file_path_rel))
    print('Number of labels: ', len(question_species))

    df_final = pd.DataFrame({'file_path': file_path_rel, 'species': question_species})
    num_species, name_species = pd.factorize(df_final['species'])
    df_final['encoded_species'] = num_species
    file_name = os.path.join(base_dir, "train_v3.csv")
    df_final.to_csv(file_name, columns=['file_path', 'encoded_species'], sep=',', index=False)
    label_file_name = os.path.join(base_dir, "label_to_species_V3.csv")
    df_labels = df_final.groupby(by=['species', 'encoded_species'], as_index=False).first()
    df_labels.to_csv(label_file_name, columns=['species', 'encoded_species'], sep=',', index=False)
