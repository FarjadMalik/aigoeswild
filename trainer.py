"""

"""
# importing the modules
import os
import logging
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score

import config
import models
from data_loader.prepare_data import generate_train_dataset


def train():
    tf.random.set_seed(config.seed)
    os.makedirs(os.path.dirname(config.logs_dir), exist_ok=True)
    logs_filename = os.path.join(config.logs_dir, config.logs_filename)
    # now we will Create and configure logger
    logging.basicConfig(filename=logs_filename,
                        format='%(asctime)s %(message)s',
                        filemode='w')

    # Let us Create an object
    logger = logging.getLogger()

    # Now we are going to Set the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)

    # # some messages to test
    # logger.debug("This is just a harmless debug message")
    # logger.info("This is just an information for you")
    # logger.warning("OOPS!!!Its a Warning")
    # logger.error("Have you try to divide a number by zero")
    # logger.critical("The Internet is not working....")

    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')

    logger.info(f"...\n")
    logger.info(f"GPU: \n{gpus}")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # get the train and test datasets
    train_ds, valid_ds = generate_train_dataset()
    logger.info(f"Length train set: \n{len(train_ds)}")
    logger.info(f"Length valid set: \n{len(valid_ds)}")

    logger.info(f"load_pretrained_imagenet_resnet18: \n{config.load_pretrained_imagenet_resnet18}")

    # create model
    if config.load_pretrained_imagenet_resnet18:
        # model = models.load_pretrained_imagenet_resnet18_model()
        model = None
        pass
    else:
        model = models.get_model()
    logger.info(f"Model Summary: \n{model.summary()}")
    # define loss and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    #  and other variables to keep a track of our total loss and accuracy
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    top1_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='top1_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='top1_val_accuracy')

    @tf.function
    def train_step(t_images, t_labels):
        with tf.GradientTape() as tape:
            predictions = model(t_images, training=True)
            loss = loss_object(y_true=t_labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss(loss)
        top1_accuracy(t_labels, predictions)

        return predictions, loss

    @tf.function
    def valid_step(v_images, v_labels):
        predictions = model(v_images, training=False)
        v_loss = loss_object(v_labels, predictions)

        valid_loss(v_loss)
        valid_accuracy(v_labels, predictions)
        return predictions, v_loss

    # Loading the previously saved weights
    if config.load_model_dir != "":
        model.load_weights(filepath=config.load_model_dir)

    logger.info(f"two_phase_training: \n{config.two_phase_training}")

    if config.two_phase_training:
        # freezing layers in case of two-phase training
        # also if more layers need to be added at the end
        model.summary()
        fine_tune_at = 8
        for layer in model.layers[:fine_tune_at]:
            layer.trainable = False
        print(f"Model Summary phase 2: \n")
        model.summary()
        print(f"Trainable variables in the phase 2 model: {len(model.trainable_variables)}")
        print(f"Trainable weights in the phase 2 model: {len(model.trainable_weights)}")
        logger.info(f"Trainable variables in the phase 2 model: {len(model.trainable_variables)}")

    n_iterations = len(train_ds)
    print(f"_________________________________________________________________")
    print(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    print(f"Training set iterations: {n_iterations}")
    logger.info(f"Training set iterations: {n_iterations}")
    print(f"Training...")
    logger.info(f"Training...")
    logger.info(f"_________________________________________________________________")
    print(f"_________________________________________________________________")

    # start training
    for epoch in range(config.start_epoch, config.EPOCHS):
        # reset states
        train_loss.reset_states()
        top1_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()
        print(f"Epoch {epoch+1}:")

        # Empty lists to store predictions and true labels
        true_labels = []
        predicted_labels = []

        step = 0
        # one full iteration of the train dataset
        for filenames, images, labels in train_ds:
            step = step + 1
            _, _ = train_step(images, labels)

            if step % 100 == 0:
                print(f"Epoch: {epoch + 1}/{config.EPOCHS}, step:{step}/{n_iterations},"
                      f" train loss: {train_loss.result():.5f}, train top 1 accuracy: {top1_accuracy.result():.5f}")

                # logger.info(f"Epoch: {epoch + 1}/{config.EPOCHS}, step:{step}/{n_iterations},"
                #             f" train loss: {train_loss.result():.5f}, train top 1 accuracy: "
                #             f"{top1_accuracy.result():.5f}")

        # run a validation step
        for filenames, valid_images, valid_labels in valid_ds:
            val_predictions, _ = valid_step(valid_images, valid_labels)
            # Read softmax predictions and calculate y_labels to compute f1 score
            val_pred = (val_predictions > 0.5)
            # Get predicted labels for each specie in batch
            y_val_labels = val_pred.numpy().argmax(axis=1)
            # add these to a list to calculate the overall scores at the end
            true_labels.extend(valid_labels)
            predicted_labels.extend(y_val_labels)

        # Calculate scores of predictions
        f1 = f1_score(true_labels, predicted_labels, average='macro')
        precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
        recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)

        # logging results
        print(f"Epoch completed: {epoch + 1}/{config.EPOCHS},"
              f" train loss: {train_loss.result():.5f}, train top 1 accuracy: {top1_accuracy.result():.5f},\n"
              f" valid loss: {valid_loss.result():.5f}, valid top 1 accuracy: {valid_accuracy.result():.5f},\n"
              f" F1 score: {f1:.5f}, Precision: {precision:.5f}, Recall: {recall: .5f}")

        logger.info(f"Epoch completed: {epoch + 1}/{config.EPOCHS},"
                    f" train loss: {train_loss.result():.5f}, train top 1 accuracy: {top1_accuracy.result():.5f},\n"
                    f" valid loss: {valid_loss.result():.5f}, valid top 1 accuracy: {valid_accuracy.result():.5f},\n"
                    f" F1 score: {f1:.5f}, Precision: {precision:.5f}, Recall: {recall: .5f}")

        # Model finished training after all epochs. Saving the model
        # Check if the path to the file exists.
        os.makedirs(os.path.dirname(config.save_model_dir), exist_ok=True)
        # save the latest model in the main folder
        model.save_weights(filepath=config.save_model_dir, save_format='tf')

        # Save current epoch with its epoch number for callback as well
        saved_models_list = config.save_model_dir.split('/')
        epoch_dir = os.path.join(str(saved_models_list[0]), str(saved_models_list[1]), str(epoch+1))
        model.save_weights(filepath=epoch_dir, save_format='tf')

    print(f"---End of training---")
