"""

"""
import keras.backend as K
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score

import config
import models
from data_loader.prepare_data import generate_train_dataset


def f1_score_imbalanced(y_label, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_label * y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_label)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_label) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_label * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def evaluate():
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # get the train and test datasets
    train_ds, test_ds = generate_train_dataset()

    # load the model
    model = models.get_model()
    # Loading the previously saved weights
    model.load_weights(filepath=config.save_model_dir)

    # Get the accuracy on the test set
    loss_object = tf.keras.metrics.SparseCategoricalCrossentropy()
    test_loss = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    print(f"Tensor flow mode: {tf.executing_eagerly()}")

    @tf.function
    def test_step(test_images, test_labels):
        s_predictions = model(test_images, training=False)
        s_loss = loss_object(test_labels, s_predictions)

        test_loss(s_loss)
        test_accuracy(test_labels, s_predictions)

        return s_predictions, s_loss

    # Empty lists to store predictions and true labels
    true_labels = []
    predicted_labels = []
    print(f'Iterating test data:')
    for filenames, images, labels in test_ds:
        predictions, t_loss = test_step(images, labels)
        y_pred = (predictions > 0.5)
        y_labels = y_pred.numpy().argmax(axis=1)

        # add these to the list to calculate the overall scores at the end
        true_labels.extend(labels)
        predicted_labels.extend(y_labels)

    print(f":done")
    print(f"True Labels: {true_labels}")
    print(f"Model Predictions: {predicted_labels}")
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    # f1_imbalanced = f1_score_imbalanced(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    print(f"F1 score: {f1}")
    # print(f"F1 score with custom imbalanced func: {f1_imbalanced}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    print("The accuracy on test set is: {:.3f}%".format(test_accuracy.result() * 100))
