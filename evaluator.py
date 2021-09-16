"""

"""
import os
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report

import config
import models
from data_loader.prepare_data import generate_test_dataset


def evaluate():
    tf.random.set_seed(config.seed)
    # Create logs dir if not already exists
    os.makedirs(os.path.dirname(config.logs_dir), exist_ok=True)
    # Configuring logger
    logs_filename = os.path.join(config.logs_dir, config.logs_filename)
    logging.basicConfig(filename=logs_filename,
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Fetch GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    logger.info(f"...\n")
    logger.info(f"GPU: \n{gpus}")

    # Get the model
    model = models.get_model()
    # Loading the previously saved weights
    model.load_weights(filepath=os.path.join(os.path.dirname(__file__), config.load_model_dir))
    logger.info(f"Model Summary: \n{model.summary()}")

    load_model_dir = config.load_model_dir.split('\\')
    load_model_name = config.model_type
    load_model_epoch = load_model_dir[2]
    print(f"Running test on model: {load_model_name} Epoch {load_model_epoch}")
    logger.info(f"Running test on model: {load_model_name} Epoch {load_model_epoch}")
    logger.info(f"...\n")

    # get the test datasets
    test_ds = generate_test_dataset()

    # Get the accuracy on the test set
    loss_object = tf.keras.metrics.SparseCategoricalCrossentropy()
    test_loss = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    # Define a test step which iterates over the batch
    # and returns the predicted softmax outputs
    @tf.function
    def test_step(test_images, test_labels):
        s_predictions = model(test_images, training=False)
        s_loss = loss_object(test_labels, s_predictions)

        test_loss(s_loss)
        test_accuracy(test_labels, s_predictions)

        return s_predictions, s_loss

    # Create empty lists to store predictions and true labels
    # We use these later on to calculate per specie statistics

    true_labels = []
    predicted_labels = []
    image_paths = []

    print(f'Iterating test data:')
    print(f"_________________________________________________________________")

    for filenames, images, labels in test_ds:
        predictions, t_loss = test_step(images, labels)
        y_pred = (predictions > 0.5)
        y_labels = y_pred.numpy().argmax(axis=1)

        # add these to the list to calculate the overall scores at the end
        image_paths.extend(filenames.numpy())
        true_labels.extend(labels.numpy())
        predicted_labels.extend(y_labels)

    print(f":done")
    logger.info(f":done")
    # Inference loops is finished now we can start calculating the rest of the statistics

    # Sklearn metrics
    precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    cm = confusion_matrix(true_labels, predicted_labels)
    cr = classification_report(true_labels, predicted_labels, digits=3, zero_division=0, output_dict=True)

    # Read specie mapping table to get results per specie name (instead of label)
    species_mapping = pd.read_csv(config.file_specie_mapping)
    # Read number of images of each specie used for training
    count_species = pd.read_csv(config.file_train_count_specie)
    # Read number of images of each specie used for training
    test_count_species = pd.read_csv(config.file_test_count_specie)

    logger.info(f"Specie mapping file to be used: {config.file_specie_mapping}")
    logger.info(f"Count species for train: {config.file_train_count_specie}")
    logger.info(f"Count species for test: {config.file_test_count_specie}")

    # From classification report we create dataframe and remove unnecessary rows
    # columns needed = specie, precision, recall, f1score, accuracy, number_of_images
    prediction_report = pd.DataFrame(cr).transpose()
    # remove last rows containing micro and macro stats (not per label)
    prediction_report = prediction_report[prediction_report.index.astype('str').str.isnumeric()]
    # index is basically our class label
    prediction_report['class'] = prediction_report.index
    prediction_report['class'] = prediction_report['class'].apply(lambda x: x if x.isnumeric() else 99)
    prediction_report['class'] = prediction_report['class'].astype(np.int32)

    # we can use confusion matrix diagonal to report accuracy of each class
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # The diagonal entries are the accuracies of each class
    # Accuracy per specie is the same as recall so can be ignored
    # prediction_report['accuracy'] = cm.diagonal()

    # Join with specie mapping to get specie names
    prediction_report = pd.merge(left=prediction_report, right=species_mapping, how="inner", left_on="class",
                                 right_on="encoded_species")
    # del unnecessary columns
    del prediction_report["encoded_species"]
    del prediction_report["class"]
    del prediction_report["support"]

    # Rearrange columns. Put species as the first column
    cols = prediction_report.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    prediction_report = prediction_report.reindex(columns=cols)

    # Merge with number of images for each specie used for training
    prediction_report = pd.merge(left=prediction_report, right=count_species, how="inner", left_on="question__species",
                                 right_on="question__species")
    # Merge with number of images for each specie used for training
    prediction_report = pd.merge(left=prediction_report, right=test_count_species, how="inner",
                                 left_on="question__species", right_on="question__species")
    prediction_report.columns = ['Species', 'Precision', 'Recall', 'F1-score', 'Count (Train)',
                                 'Count (Test)']
    # sorted by number of images used for train
    prediction_report = prediction_report.sort_values('Count (Train)', ascending=False)

    # The statistics are ready. Lets log and save these to output files
    print("The accuracy on test set is: {:.3f}%".format(test_accuracy.result() * 100))
    print(f"The accuracy on test set is: {test_accuracy.result(): .5f}")
    print(f"F1 score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Confusion Matrix: \n{cm}")
    print(f"Classification report: \n{prediction_report}")
    # Log file info
    logger.info("The accuracy on test set is: {:.3f}%".format(test_accuracy.result() * 100))
    logger.info(f"The accuracy on test set is: {test_accuracy.result(): .5f}")
    logger.info(f"F1 score: {f1}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")

    # Storing results. check if output directory already exists
    os.makedirs(config.results_output_dir, exist_ok=True)
    # Create output file names
    fn_results = f"{load_model_name}_{load_model_epoch}_test_predictions.csv"
    fn_predicted_report = f"{load_model_name}_{load_model_epoch}_test_per_specie_report.csv"
    # Create a predictions df with each image and its true and predicted class
    results_df = pd.DataFrame(list(zip(image_paths, true_labels, predicted_labels)),
                              columns=['image_path', 'true_label', 'predicted_label'])
    # Store this as a csv
    results_df.to_csv(os.path.join(config.results_output_dir, fn_results), sep=',', header=True, index=False)
    prediction_report.to_csv(os.path.join(config.results_output_dir, fn_predicted_report), sep=',', header=True,
                             index=False)


if __name__ == '__main__':
    print(f'---Major Tom to Ground Control---')
    evaluate()
    print(f'------')
