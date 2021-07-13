"""

"""
import os
import cv2
import pandas as pd

if __name__ == "__main__":
    path_csv = r'C:\Users\mfarj\Documents\ss_data\data_csv\test_input.csv'
    path_csv_2 = r'C:\Users\mfarj\Documents\ss_data\data_csv\valid_input.csv'
    train_df = pd.read_csv(path_csv)
    train_df_2 = pd.read_csv(path_csv_2)
    train_df = train_df.append(train_df_2)
    print(len(train_df))
    base_dir = r'C:\Users\mfarj\Documents\ss_data\snapshotserengeti-unzipped\snapshotserengeti-unzipped'
    train_df['file_path_local'] = train_df.image_path_rel.map(lambda x: os.path.join(base_dir, x))

    for index, file_name in enumerate(train_df.file_path_local):
        # index = start_from + index
        # if index == 5:
        #     print(f'Breaking at size {index}')
        #     break

        image = cv2.imread(file_name)
        if image is not None:
            height, width, channels = image.shape
            if height != 256 or width != 256:
                print(f"Resizing {file_name}")
                print(f"Original shape {image.shape}")
                image = cv2.resize(image, (256, 256))
                cv2.imwrite(file_name, image)
        else:
            print(f"File not found: {file_name}")
        if index % 10000 == 0:
            print(index)
        if index % 100000 == 0:
            print(index)

    print(f"---End---")
