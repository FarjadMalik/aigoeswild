"""

"""
import os

import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score

import config
import models
from data_loader.prepare_data import generate_train_dataset


def train():
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # get the train and test datasets
    train_ds, valid_ds = generate_train_dataset()

    # create model
    model = models.get_model()
    # define loss and optimizer and other variables to keep a track of our total loss and accuracy
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    top1_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='top1_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='top1_val_accuracy')
    # top5_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')

    @tf.function
    def train_step(t_images, t_labels):
        with tf.GradientTape() as tape:
            predictions = model(t_images, training=True)
            loss = loss_object(y_true=t_labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss(loss)
        top1_accuracy(t_labels, predictions)
        # top5_accuracy(t_labels, predictions)

        return predictions, loss

    @tf.function
    def valid_step(v_images, v_labels):
        predictions = model(v_images, training=False)
        v_loss = loss_object(v_labels, predictions)

        valid_loss(v_loss)
        valid_accuracy(v_labels, predictions)
        # top5_accuracy(v_labels, predictions)
        return predictions, v_loss

    n_iterations = len(train_ds)
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Training set iterations: {n_iterations}")
    print(f"Training...")

    # start training
    for epoch in range(config.EPOCHS):
        train_loss.reset_states()
        top1_accuracy.reset_states()
        # valid_loss.reset_states()
        # valid_accuracy.reset_states()

        # Empty lists to store predictions and true labels
        true_labels = []
        predicted_labels = []

        step = 0
        # one full iteration of the train dataset
        for filenames, images, labels in train_ds:
            step = step + 1
            t_predictions, _ = train_step(images, labels)
            # Calculating f1 score for train set
            # # Get predicted labels for each specie in batch
            # y_pred = (t_predictions > 0.5)
            # y_labels = y_pred.numpy().argmax(axis=1)
            # # add these to a list to calculate the overall scores at the end
            # true_labels.extend(labels)
            # predicted_labels.extend(y_labels)
            if step % 10 == 0:
                print(f"Epoch: {epoch + 1}/{config.EPOCHS}, step:{step}/{n_iterations},"
                      f" train loss: {train_loss.result():.5f}, top 1 accuracy: {top1_accuracy.result():.5f}")

        # run a validation step
        for filenames, valid_images, valid_labels in valid_ds:
            val_predictions, _ = valid_step(valid_images, valid_labels)
            # Read softmax predictions and calculate y_labels to compute f1 score
            # Get predicted labels for each specie in batch
            val_pred = (val_predictions > 0.5)
            y_val_labels = val_pred.numpy().argmax(axis=1)
            # add these to a list to calculate the overall scores at the end
            true_labels.extend(valid_labels)
            predicted_labels.extend(y_val_labels)

        # Calculate scores of predictions
        f1 = f1_score(true_labels, predicted_labels, average='macro')
        precision = precision_score(true_labels, predicted_labels, average='macro')
        recall = recall_score(true_labels, predicted_labels, average='macro')
        # logging results
        print(f"Epoch completed: {epoch + 1}/{config.EPOCHS},"
              f" train loss: {train_loss.result():.5f}, top 1 accuracy: {top1_accuracy.result():.5f},\n"
              f" valid loss: {valid_loss.result():.5f}, valid top 1 accuracy: {valid_accuracy.result():.5f},\n"
              f" F1 score: {f1:.5f}, Precision: {precision:.5f}, Recall: {recall: .5f}")

        # Model finished training after all epochs. Saving the model
        # Check if the path to the file exists. If it doesnt create it using os make dir
        os.makedirs(os.path.dirname(config.save_model_dir), exist_ok=True)
        model.save_weights(filepath=config.save_model_dir, save_format='tf')

    print(f"---End of training---")
