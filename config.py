"""

"""
# Runner mode - Can be either train or evaluate
mode = 'train'  # evaluate

# Training Parameters
EPOCHS = 55
BATCH_SIZE: int = 128
NUM_CLASSES = 55
image_height = 224
image_width = 224
channels = 3
seed = 42

# Model Parameters
save_model_dir = "saved_models/species_only_singles/model"

# choose a network
model = "resnet18"
# model = "resnet34"
# model = "resnet50"
# model = "resnet101"
# model = "resnet152"


# Data Parameters
train_csv = r"C:\Users\mfarj\Documents\ss_data\train_species_only_singles.csv"
data_dir = r"C:\Users\mfarj\Documents\ss_data\snapshotserengeti-unzipped\snapshotserengeti-unzipped"
