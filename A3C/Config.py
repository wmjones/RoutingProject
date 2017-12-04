import multiprocessing


class Config:
    # AGENTS = multiprocessing.cpu_count()
    AGENTS = 4
    PREDICTORS = 1
    TRAINERS = 1
    DEVICE = 'cpu:0'

    PREDICTION_BATCH_SIZE = 10
    LEARNING_RATE = 1e-3
    TRAINING_MIN_BATCH_SIZE = 10
    RESTORE = False
    NUM_OF_CUSTOMERS = 8
    LAYERS_STACKED_COUNT = 2
    RNN_HIDDEN_DIM = 10
    DNN_HIDDEN_DIM = 10
    OR_TOOLS = True
