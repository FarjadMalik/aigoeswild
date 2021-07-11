"""

"""
# Runner mode - Can be either train or evaluate
mode = 'train'  # train/evaluate

# Training Parameters
EPOCHS = 55
BATCH_SIZE: int = 128
NUM_CLASSES = 55
image_height = 224
image_width = 224
channels = 3
seed = 42

# Model Parameters
save_model_dir = "saved_models/res18_species_custom_split_capture_ids_20210711/model"

# choose a network
model = "resnet18"
# model = "resnet34"
# model = "resnet50"
# model = "resnet101"
# model = "resnet152"


# Data Parameters
train_csv = r"C:\Users\mfarj\Documents\ss_data\data_csv\train_phase2_split.csv"
test_csv = r"C:\Users\mfarj\Documents\ss_data\data_csv\test_phase2_split.csv"
data_dir = r"C:\Users\mfarj\Documents\ss_data\snapshotserengeti-unzipped\snapshotserengeti-unzipped"
