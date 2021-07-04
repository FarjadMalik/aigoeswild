"""

"""
import tensorflow as tf
import models
import config
from data_loader.prepare_data import generate_datasets


def evaluate():

    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # get the train and test datasets
    filenames_train, labels_train, train_ds, filenames_test, labels_test, test_ds = generate_datasets()

    # load the model
    model = models.get_model()
    model.load_weights(filepath=config.save_model_dir)

    # Get the accuracy on the test set
    loss_object = tf.keras.metrics.SparseCategoricalCrossentropy()
    test_loss = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def test_step(test_images, test_labels):
        predictions = model(test_images, training=False)
        t_loss = loss_object(test_labels, predictions)

        test_loss(t_loss)
        test_accuracy(test_labels, predictions)

    for filenames, images, labels in test_ds:
        test_step(images, labels)
        print("loss: {:.5f}, test accuracy: {:.5f}".format(test_loss.result(),
                                                           test_accuracy.result()))

    print("The accuracy on test set is: {:.3f}%".format(test_accuracy.result() * 100))
