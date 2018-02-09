import multiprocessing


class Config:
    # AGENTS = multiprocessing.cpu_count()
    AGENTS = 4
    PREDICTORS = 1
    TRAINERS = 1
    DEVICE = 'cpu:0'

    MAX_BATCH_SIZE = 20
    PREDICTION_BATCH_SIZE = 20
    LEARNING_RATE = 1e-3
    TRAINING_MIN_BATCH_SIZE = 15
    RESTORE = False
    NUM_OF_CUSTOMERS = 8
    LAYERS_STACKED_COUNT = 2
    RNN_HIDDEN_DIM = 128
    DNN_HIDDEN_DIM = 15
    OR_TOOLS = True
    RUN_TIME = 10

    DIRECTION = 5
    CELL_TYPE = 0
    TRAIN = False
    # GPU = True
