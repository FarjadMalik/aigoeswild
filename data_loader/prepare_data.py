import tensorflow as tf
import config
import os
import numpy as np

from sklearn.model_selection import train_test_split


def filter_valid_filenames(filenames, labels):
    # remove files which are not jpeg, not downloaded correctly
    true_files = []
    true_labels = []
    i = 0
    for f in filenames:
        if os.path.isfile(f) and os.stat(f).st_size > 0:
            true_files.append(f)
            true_labels.append(labels[i])
        i += 1
    return true_files, true_labels


def read_inputs(train_csv, data_dir):
    print('Reading dataset')

    # Read input csv with file paths and the encoded species
    file_paths, labels = _read_label_file(train_csv, ',')
    # join with the base dir to get full path of the image
    filenames = [os.path.join(data_dir, i.replace('/', '\\')) for i in file_paths]
    image_count = len(filenames)
    print(f'Count filenames: {image_count}')

    # Filtering not needed as i already do this as part of input csv generation
    # filenames, labels = filter_valid_filenames(filenames, labels)
    # image_count = len(filenames)
    # print('Count filenames after filtering:', image_count)

    # Return filenames and their corresponding labels
    return filenames, labels


def _read_label_file(file, delimiter):
    f = open(file, "r")
    next(f)
    image_names = []
    labels = []
    for line in f:
        tokens = line.split(delimiter)
        # Token[0] = file_path_rel
        image_names.append(tokens[0])
        # Token[1] = encoded specie label
        labels.append(tf.cast(int(tokens[1]), tf.int64))
    return image_names, labels


def decode_img(img, crop_size, num_channels):
    img = tf.image.decode_jpeg(img, channels=num_channels)  # color images
    img = tf.image.convert_image_dtype(img, tf.float32)
    # convert unit8 tensor to floats in the [0,1]range
    return tf.image.resize(img, crop_size)


def _read_images(filename, label):
    img = tf.io.read_file(filename)
    try:
        img = decode_img(img=img, crop_size=[config.image_height, config.image_width],
                         num_channels=config.channels)
    except Exception as e:
        print(f'_read_images, Corrupted file: {filename} with exception as {e}')

    return filename, img, label


def generate_datasets():
    filenames, labels = read_inputs(config.train_csv, config.data_dir)

    filenames_train, filenames_test, labels_train, labels_test = train_test_split(filenames,
                                                                                  labels,
                                                                                  stratify=labels,
                                                                                  test_size=0.2,
                                                                                  random_state=config.seed)

    print(f'Train filenames Count: {len(filenames_train)}')
    print(f'Test filenames Count: {len(filenames_test)}')

    # Count frequency of each species in the test and train datasets
    print(f"Specie Frequencies in train dataset: \n"
          f"{list(zip(*np.unique(labels_train, return_counts=True)))}")
    print(f"Specie Frequencies in test dataset: \n"
          f"{list(zip(*np.unique(labels_test, return_counts=True)))}")

    # todo: remove species with less than x images (x being 10, 50, 100?)

    # Load this on a tensor
    train_ds = tf.data.Dataset.from_tensor_slices((filenames_train, labels_train))
    test_ds = tf.data.Dataset.from_tensor_slices((filenames_test, labels_test))

    # Read the images on the tensors as well along with the filenames and the labels
    train_ds = train_ds.map(_read_images)
    test_ds = test_ds.map(_read_images)

    train_ds.shuffle(buffer_size=len(labels_train))
    test_ds.shuffle(buffer_size=len(labels_test))

    # read the original_dataset in the form of batch
    train_ds = train_ds.shuffle(buffer_size=1000).batch(batch_size=config.BATCH_SIZE)
    test_ds = test_ds.batch(batch_size=config.BATCH_SIZE)

    return train_ds, test_ds
