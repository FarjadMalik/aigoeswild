import os

import numpy as np
import tensorflow as tf

import config


def _train_preprocess(filename, img, label):
    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    reshaped_image = tf.image.random_crop(img, [config.image_height, config.image_width,
                                                config.channels])

    # Randomly flip the image horizontally.
    reshaped_image = tf.image.random_flip_left_right(reshaped_image)

    # # Because these operations are not commutative, consider randomizing
    # # the order their operation.
    # reshaped_image = tf.image.random_brightness(reshaped_image, max_delta=63)
    # # Randomly changing contrast of the image
    # reshaped_image = tf.image.random_contrast(reshaped_image, lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    reshaped_image = tf.image.per_image_standardization(reshaped_image)

    # Set the shapes of tensors.
    reshaped_image.set_shape([config.image_height, config.image_width, config.channels])

    return filename, img, label


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
    # Read input csv with file paths and the encoded species
    file_paths, labels = _read_label_file(train_csv, ',')
    # join with the base dir to get full path of the image
    filenames = [os.path.join(data_dir, i.replace('/', '\\')) for i in file_paths]
    # Filtering not needed as i already do this as part of input csv generation
    # filenames, labels = filter_valid_filenames(filenames, labels)
    # image_count = len(filenames)
    # print('Count filenames after filtering:', image_count)

    # Return filenames and their corresponding labels
    return filenames, labels


def _read_label_file(file, delimiter):
    # index = 0
    f = open(file, "r")
    next(f)
    image_names = []
    labels = []
    for line in f:
        tokens = line.split(delimiter)
        # Token[0] = image_path_rel
        image_names.append(tokens[0])
        # Token[1] = encoded specie label
        labels.append(tf.cast(int(tokens[1]), tf.int64))
        # index += 1
        # if index == 400:
        #     break
    return image_names, labels


def decode_img(img, num_channels):
    img = tf.image.decode_jpeg(img, channels=num_channels)  # color images
    # convert unit8 tensor to floats in the [0,1]range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Resize image: not needed
    # img = tf.image.resize(img, [256, 256])
    return img


def _read_images(filename, label):
    img = tf.io.read_file(filename)
    try:
        img = decode_img(img=img, num_channels=config.channels)
    except Exception as e:
        print(f'_read_images, Corrupted file: {filename} with exception as {e}')

    return filename, img, label


def generate_test_dataset():
    filenames_test, labels_test = read_inputs(config.test_csv, config.data_dir)

    print(f'Test filenames Count: {len(filenames_test)}')

    # Count frequency of each species in the test and train datasets
    print(f"Specie Frequencies in test dataset: \n"
          f"{list(zip(*np.unique(labels_test, return_counts=True)))}")

    # Load this on a tensor
    test_ds = tf.data.Dataset.from_tensor_slices((filenames_test, labels_test))

    # Read the images on the tensors as well along with the filenames and the labels
    test_ds = test_ds.map(_read_images)
    test_ds.shuffle(buffer_size=len(labels_test))

    # read the original_dataset in the form of batch
    test_ds = test_ds.batch(batch_size=config.BATCH_SIZE)

    return test_ds


def generate_train_dataset():
    # Read inputs
    filenames_train, labels_train = read_inputs(config.train_csv, config.data_dir)
    filenames_valid, labels_valid = read_inputs(config.valid_csv, config.data_dir)

    print(f'Train filenames Count: {len(filenames_train)}')
    print(f'Validation filenames Count: {len(filenames_valid)}')

    # Count frequency of each species in the test and train datasets
    print(f"Specie Frequencies in train dataset: \n"
          f"{list(zip(*np.unique(labels_train, return_counts=True)))}")
    print(f"Specie Frequencies in validation dataset: \n"
          f"{list(zip(*np.unique(labels_valid, return_counts=True)))}")

    # Load this on a tensor
    train_ds = tf.data.Dataset.from_tensor_slices((filenames_train, labels_train))
    valid_ds = tf.data.Dataset.from_tensor_slices((filenames_valid, labels_valid))

    # Read the images on the tensors as well along with the filenames and the labels
    train_ds = train_ds.map(_read_images)
    valid_ds = valid_ds.map(_read_images)
    train_ds = train_ds.map(_train_preprocess)

    train_ds.shuffle(buffer_size=len(labels_train), seed=config.seed)
    valid_ds.shuffle(buffer_size=len(labels_valid), seed=config.seed)

    # read the original_dataset in the form of batch
    train_ds = train_ds.batch(batch_size=config.BATCH_SIZE)
    valid_ds = valid_ds.batch(batch_size=config.BATCH_SIZE)

    return train_ds, valid_ds
