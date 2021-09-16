"""

"""
# Runner mode - Can be either train or evaluate
mode = 'train'  # train/evaluate

# Training Parameters
EPOCHS = 15
BATCH_SIZE: int = 64
NUM_CLASSES = 52
image_height = 224
image_width = 224
channels = 3
seed = 12345

# is two phase training check to freeze layers in that case
two_phase_training = False

# Base model imagenet weights
load_pretrained_imagenet_resnet18 = False

# Model Parameters
load_model_dir = ""
# r"saved_models\res18_adam_baseline_v2\9"
start_epoch = 0
# total 8 epochs
save_model_dir = r"saved_models/res18_adam_balancedphase1_12345/model"

results_output_dir = "results"

# best model yet: saved_models/base1_res18_transfer_imagenet_adam_epoch9_best/model
# ""
# "saved_models/res18_adam_ros_rus/model"

# choose a network
model = "resnet18"
# model = "resnet34"
# model = "resnet50"
# model = "resnet101"
# model = "resnet152"


# Data Parameters
# model being used. Either ROS, RUS, Baseline, Balanced(5K), ROSRUS(15K)
# and phase either 1 or 2 (except for baseline)
model_type = "Balanced_phase1_seed12345"
# For simon: Change file paths here
# og train file
# train_csv = r"C:\Users\mfarj\Documents\ss_data\data_csv\train_input.csv"
# ros + rus file
# train_csv = r"C:\Users\mfarj\Documents\ss_data\data_csv\ROS+RUS_train_input.csv"
# ros file
# train_csv = r"C:\Users\mfarj\Documents\ss_data\data_csv\ROS_train_input.csv"
# balanced file
train_csv = r"C:\Users\mfarj\Documents\ss_data\data_csv\Phase1(balanced)_train_input.csv"
# rus file
# train_csv = r"C:\Users\mfarj\Documents\ss_data\data_csv\RUS_train_input.csv"
test_csv = r"C:\Users\mfarj\Documents\ss_data\data_csv\test_input.csv"
valid_csv = r"C:\Users\mfarj\Documents\ss_data\data_csv\valid_input.csv"
data_dir = r"C:\Users\mfarj\Documents\ss_data\snapshotserengeti-unzipped\snapshotserengeti-unzipped"

# specie_mapping file
file_specie_mapping = r"C:\Users\mfarj\Documents\ss_data\data_csv\label_to_species_mapping.csv"

# count per specie file
file_train_count_specie = r"C:\Users\mfarj\Documents\ss_data\data_csv\Phase1(balanced)_train_count_per_species.csv"
file_test_count_specie = r"C:\Users\mfarj\Documents\ss_data\data_csv\test_count_per_species.csv"

# Logging configuration
logs_dir = "logs/"
logs_filename = "Balanced_phase1_seed12345_train.log"

