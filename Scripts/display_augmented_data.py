"""

"""
from data_loader.prepare_data import read_inputs

from matplotlib import pyplot as plt
import argparse
import config
import tensorflow as tf


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


def _train_preprocess(img):
    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    reshaped_image = tf.image.random_crop(img, [config.image_height, config.image_width,
                                                config.channels])

    # Randomly flip the image horizontally.
    reshaped_image = tf.image.random_flip_left_right(reshaped_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    reshaped_image = tf.image.random_brightness(reshaped_image, max_delta=63)
    # Randomly changing contrast of the image
    reshaped_image = tf.image.random_contrast(reshaped_image, lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    reshaped_image = tf.image.per_image_standardization(reshaped_image)

    # Set the shapes of tensors.
    reshaped_image.set_shape([config.image_height, config.image_width, config.channels])

    return reshaped_image


def main(input_csv, data_dir):
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')

    print(f"...\n")
    print(f"GPU: \n{gpus}")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Read inputs
    filenames_train, labels_train = read_inputs(input_csv, data_dir)

    train_ds = tf.data.Dataset.from_tensor_slices((filenames_train, labels_train))

    # Read the images on the tensors as well along with the filenames and the labels
    train_ds = train_ds.map(_read_images)
    # read the original_dataset in the form of batch
    train_ds = train_ds.shuffle(buffer_size=1000)

    # Set the figure size - handy for larger output
    plt.rcParams["figure.figsize"] = [6, 12]

    filename, image, label = next(iter(train_ds))

    plt.figure(figsize=(10, 10))
    for i in range(9):
        reshaped_image = _train_preprocess(image)
        plt.imshow(reshaped_image)
        ax = plt.subplot(3, 3, i + 1)
        plt.axis("off")

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create_species_only_file_and_resize.py')
    # Add parser arguments
    parser.add_argument("--input-csv", default=r"C:\Users\mfarj\Documents\ss_data\data_csv\data_aug_script_input.csv",
                        type=str,
                        help="Path to the input image csv")
    parser.add_argument("--data-dir", default=r"C:\Users\mfarj\Documents\ss_data\snapshotserengeti-unzipped"
                                              r"\snapshotserengeti-unzipped",
                        type=str,
                        help="Path to the input image csv")
    # Execute the parse_args() method
    args = parser.parse_args()
    print(f'---Major Tom to Ground Control---')
    main(args.input_csv, args.data_dir)
    print(f'------')
