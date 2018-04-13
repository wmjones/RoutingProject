class Config:
    # AGENTS = multiprocessing.cpu_count()
    AGENTS = 4
    PREDICTORS = 1
    TRAINERS = 1
    DEVICE = 'cpu:0'

    PREDICTION_BATCH_SIZE = 20
    LEARNING_RATE = 1e-4
    TRAINING_MIN_BATCH_SIZE = 20
    NUM_OF_CUSTOMERS = 19
    LAYERS_STACKED_COUNT = 2
    RNN_HIDDEN_DIM = 128
    DNN_HIDDEN_DIM = 15
    OR_TOOLS = 1
    RUN_TIME = 10

    DIRECTION = 3
    ENC_EMB = 0
    DEC_EMB = 0
    CELL_TYPE = 0
    TRAIN = 0
    GPU = 1
    MODEL_NAME = ''
    RESTORE = 0
    DROPOUT = 0
    LOGIT_CLIP_SCALAR = 10.0
    LOGIT_PENALTY = 1e6
    MAX_GRAD = 2.0
    PATH = "./"
    REINFORCE = 1
    SOFTMAX_TEMP = 1.0
    GREEDY = 1
