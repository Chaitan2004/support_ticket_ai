# config.py

# Model configuration
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 5   # Billing, Technical, Account, Feature, Complaint
MAX_LENGTH = 128

# Training configuration
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3

# Data paths
TRAIN_DATA_PATH = "data/train.csv"
VAL_DATA_PATH = "data/val.csv"

# Device configuration
DEVICE = "cpu"  # change to "cpu" if no GPU
