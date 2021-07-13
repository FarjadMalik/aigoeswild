"""

"""
# Runner mode - Can be either train or evaluate
mode = 'train'  # train/evaluate

# Training Parameters
EPOCHS = 5
BATCH_SIZE: int = 64
NUM_CLASSES = 52
image_height = 224
image_width = 224
channels = 3
seed = 42

# Model Parameters
save_model_dir = "saved_models/base1_res18_species_64bat_5epoch_aug/model"

# choose a network
model = "resnet18"
# model = "resnet34"
# model = "resnet50"
# model = "resnet101"
# model = "resnet152"


# Data Parameters
# For simon: Change file paths here
train_csv = r"C:\Users\mfarj\Documents\ss_data\data_csv\train_input.csv"
test_csv = r"C:\Users\mfarj\Documents\ss_data\data_csv\test_input.csv"
valid_csv = r"C:\Users\mfarj\Documents\ss_data\data_csv\valid_input.csv"
data_dir = r"C:\Users\mfarj\Documents\ss_data\snapshotserengeti-unzipped\snapshotserengeti-unzipped"
