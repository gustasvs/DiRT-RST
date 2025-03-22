import torch

# game should be in this width at the top left corner of the screen
SOURCE_WIDTH = 1024
SOURCE_HEIGHT = 768

# the input to the model will be resized to this
MODEL_INPUT_WIDTH = 256
MODEL_INPUT_HEIGHT = 256

TARGET_CLASS_COUNT = 7
# file_name = 'RALLYCROSS_hellnorway' #bigger_model
file_name = 'wales_bronfelen' #smaller_model
t1 = 'sweden_hamra'
t2 = 'wales_bronfelen'
t3 = 'finland_kakaristo'


# DATA CREATION SETTINGS
SAMPLES_IN_ONE_FILE = 2000

GRAYSCALE = True
# uses transformer based spatiotemporal model if over 1
TEMPORAL_FRAME_WINDOW = 1

# TRAINING SETTINGS
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001


# PATHS
EXISTING_MODEL_PATH = 'model/model_data/{}.model'.format(file_name)
DATA_PATH = 'data/{}/file'.format(file_name)

# catch invalid settings
assert TEMPORAL_FRAME_WINDOW > 0, "TEMPORAL_FRAME_WINDOW must be greater than 0"
assert BATCH_SIZE > 0, "BATCH_SIZE must be greater than 0"
assert LEARNING_RATE > 0, "LEARNING_RATE must be greater than 0"
assert EPOCHS > 0, "EPOCHS must be greater than 0"
