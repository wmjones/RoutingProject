class Config:                   # default settings for all config options
    CELL_TYPE = 0
    BEAM_WIDTH = 5
    DEVICE = 'cpu:0'
    DIRECTION = 1
    DNN_HIDDEN_DIM = 15
    DROPOUT = 0
    FROM_FILE = 0
    GPU = 1
    LAYERS_STACKED_COUNT = 2
    LEARNING_RATE = 1e-3
    LR_DECAY_OFF = 0
    LOGIT_CLIP_SCALAR = 0
    LOGIT_PENALTY = 1e9
    MAX_GRAD = 2.0
    MOVING_AVERAGE = 0
    MODEL_NAME = ''
    MODEL_SETTING = 0
    NUM_OF_CUSTOMERS = 20
    PATH = "./"
    PREDICTION_BATCH_SIZE = 20
    PREDICTORS = 1
    REINFORCE = 0
    RESTORE = 0
    RNN_HIDDEN_DIM = 128
    RUN_TIME = 30
    REZA = 0
    STOCHASTIC = 0
    INVERSE_SOFTMAX_TEMP = 1.0
    STATE_EMBED = 0
    TRAINING_MIN_BATCH_SIZE = 20
    USE_OR_COST = 0
    USE_BAHDANAU = 0
    INPUT_TIME = 0
    INPUT_ALL = 0
    SAMPLING = 0
    SAME_BATCH = 0
    SEQUENCE_COST = 0
    USE_PCA = 0
    USE_PPO = 0
    NUM_PPO_EPOCH = 15
