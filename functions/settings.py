import torch

# game should be in this width at the top left corner of the screen
# SOURCE_WIDTH = 1024
# SOURCE_HEIGHT = 768
SOURCE_WIDTH = 1366
SOURCE_HEIGHT = 768

# the input to the model will be resized to this
# factor of 4 (TODO: test this with efficientnet_b4) as its original input size is 380x380
MODEL_INPUT_WIDTH = 341
MODEL_INPUT_HEIGHT = 192
# factor of 5
# MODEL_INPUT_WIDTH = 273
# MODEL_INPUT_HEIGHT = 154

TARGET_CLASS_COUNT = 7

file_name = 'greece_tsiristra_thea_2'
t1 = 'sweden_hamra'
t2 = 'wales_bronfelen'
t3 = 'finland_kakaristo'


# DATA CREATION SETTINGS
SAMPLES_IN_ONE_FILE = 2000

GRAYSCALE = True
# uses transformer based spatiotemporal model if over 1
TEMPORAL_FRAME_WINDOW = 6
# use to make model `see` deeper into the past by skipping frames
TEMPORAL_FRAME_GAP = 6

# TRAINING SETTINGS
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 10
BATCH_SIZE = 4
BATCHES_TO_AGGREGATE = 32
LEARNING_RATE = 0.001


# PATHS
EXISTING_MODEL_PATH = 'model/model_data/{}.model'.format(file_name)
DATA_PATH = 'data/{}/file'.format(file_name)

# catch invalid settings
assert TEMPORAL_FRAME_WINDOW > 0, "TEMPORAL_FRAME_WINDOW must be greater than 0"
assert BATCH_SIZE > 0, "BATCH_SIZE must be greater than 0"
assert LEARNING_RATE > 0, "LEARNING_RATE must be greater than 0"
assert EPOCHS > 0, "EPOCHS must be greater than 0"

# development settings

# set to None to use all data, useful for debugging
# MAX_DATA_SAMPLES = 1000
MAX_DATA_SAMPLES = None