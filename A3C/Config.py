class Config:
    # AGENTS = multiprocessing.cpu_count()
    AGENTS = 4
    PREDICTORS = 1
    TRAINERS = 1
    DEVICE = 'cpu:0'

    PREDICTION_BATCH_SIZE = 20
    LEARNING_RATE = 1e-3
    TRAINING_MIN_BATCH_SIZE = 20
    NUM_OF_CUSTOMERS = 15
    LAYERS_STACKED_COUNT = 2
    RNN_HIDDEN_DIM = 128
    DNN_HIDDEN_DIM = 15
    OR_TOOLS = True
    RUN_TIME = 10

    DIRECTION = 3
    ENC_EMB = True
    DEC_EMB = True
    CELL_TYPE = 0
    TRAIN = False
    GPU = False
    MODEL_TO_RESTORE = ''
    DROPOUT = False
    LOGIT_CLIP_SCALAR = 10
    LOGIT_PENALTY = 1e6
    MAX_GRAD = 0
    # PATH = "/Users/wyatt/Documents/Github_Repositories/RoutingProject/A3C/"
    PATH = "./"
