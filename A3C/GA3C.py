import sys
import configparser
from Config import Config
from Server import Server

for i in range(1, len(sys.argv)):
    x, y = sys.argv[i].split('=')
    setattr(Config, x, type(getattr(Config, x))(y))

if Config.MODEL_SETTING == 1:
    # working needs longer i doubled time
    Config.DIRECTION = 1
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    # Config.RUN_TIME = 64800
    Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 2:
#     # not working
#     Config.DIRECTION = 1
#     Config.USE_BAHDANAU = 1
#     Config.REINFORCE = 1
#     Config.RNN_HIDDEN_DIM = 16
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 64800
elif Config.MODEL_SETTING == 3:
    # working needs longer i doubled time
    # 1.5
    Config.DIRECTION = 2
    Config.USE_BAHDANAU = 1
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    # Config.RUN_TIME = 64800
    Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 4:
#     # not working
#     Config.DIRECTION = 2
#     Config.REINFORCE = 1
#     Config.USE_BAHDANAU = 1
#     Config.RNN_HIDDEN_DIM = 16
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 64800
# elif Config.MODEL_SETTING == 5:
#     # not working
#     Config.DIRECTION = 3
#     Config.FROM_FILE = 1
#     Config.USE_BAHDANAU = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     # Config.RUN_TIME = 64800
#     Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 6:
#     # not working
#     Config.DIRECTION = 3
#     Config.USE_BAHDANAU = 1
#     Config.REINFORCE = 1
#     Config.RUN_TIME = 64800
# elif Config.MODEL_SETTING == 7:
#     # not working
#     Config.DIRECTION = 4
#     Config.USE_BAHDANAU = 1
#     Config.FROM_FILE = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     # Config.RUN_TIME = 64800
#     Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 8:
#     # not working
#     Config.DIRECTION = 4
#     Config.USE_BAHDANAU = 1
#     Config.REINFORCE = 1
#     Config.RUN_TIME = 64800
elif Config.MODEL_SETTING == 9:
    # working but needs more time
    # has a really low starting length for some reason but didnt on rerun
    Config.DIRECTION = 10
    Config.STATE_EMBED = 1
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    # Config.RUN_TIME = 43200
    Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 10:
#     # not working 1.75
#     Config.DIRECTION = 10
#     Config.REINFORCE = 1
#     Config.STATE_EMBED = 1
#     Config.RUN_TIME = 43200
# elif Config.MODEL_SETTING == 11:
#     # not working
#     Config.DIRECTION = 10
#     Config.STATE_EMBED = 1
#     Config.REINFORCE = 1
#     Config.TRAINING_MIN_BATCH_SIZE = 30
#     Config.USE_OR_COST = 1
#     Config.RUN_TIME = 43200
# elif Config.MODEL_SETTING == 12:
#     # not working
#     Config.DIRECTION = 1
#     Config.STATE_EMBED = 1
#     Config.FROM_FILE = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     # Config.RUN_TIME = 43200
#     Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 13:
#     # not working
#     Config.DIRECTION = 1
#     Config.FROM_FILE = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     Config.STOCHASTIC = 1
#     # Config.RUN_TIME = 43200
#     Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 14:
#     # not working
#     Config.DIRECTION = 1
#     Config.REINFORCE = 1
#     Config.USE_OR_COST = 1
#     Config.RUN_TIME = 43200
# elif Config.MODEL_SETTING == 15:
#     # not working
#     Config.DIRECTION = 2
#     Config.FROM_FILE = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     Config.STOCHASTIC = 1
#     # Config.RUN_TIME = 43200
#     Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 16:
#     # not working
#     Config.DIRECTION = 2
#     Config.REINFORCE = 1
#     Config.USE_OR_COST = 1
#     Config.RUN_TIME = 43200
elif Config.MODEL_SETTING == 17:
    # working 2.25 but going down
    Config.DIRECTION = 3
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.STOCHASTIC = 1
    # Config.RUN_TIME = 43200
    Config.RUN_TIME = 259200
# elif Config.MODEL_SETTING == 18:
#     # not working
#     Config.DIRECTION = 3
#     Config.REINFORCE = 1
#     Config.USE_OR_COST = 1
#     Config.RUN_TIME = 43200
# elif Config.MODEL_SETTING == 19:
#     # not working
#     Config.DIRECTION = 4
#     Config.FROM_FILE = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     Config.STOCHASTIC = 1
#     Config.RUN_TIME = 43200
# elif Config.MODEL_SETTING == 20:
#     # not working
#     Config.DIRECTION = 4
#     Config.REINFORCE = 1
#     Config.USE_OR_COST = 1
#     Config.RUN_TIME = 43200
elif Config.MODEL_SETTING == 21:
    # not working but has worked best in the past wtf
    Config.DIRECTION = 10
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    # Config.RUN_TIME = 43200
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 22:
    # working 1.3 1.2 but went up at end
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 129600
    # Config.RUN_TIME = 43200
# elif Config.MODEL_SETTING == 23:
#     # not working
#     Config.DIRECTION = 10
#     Config.REINFORCE = 1
#     Config.MOVING_AVERAGE = 1
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 129600
#     # Config.RUN_TIME = 43200
# elif Config.MODEL_SETTING == 24:
#     # not working
#     Config.DIRECTION = 10
#     Config.REINFORCE = 1
#     Config.USE_OR_COST = 1
#     Config.RUN_TIME = 43200
# elif Config.MODEL_SETTING == 25:
#     # not working
#     Config.DIRECTION = 10
#     Config.REINFORCE = 1
#     Config.STATE_EMBED = 1
#     Config.RUN_TIME = 43200
# elif Config.MODEL_SETTING == 26:
#     # not working
#     Config.DIRECTION = 10
#     Config.REINFORCE = 1
#     Config.STATE_EMBED = 1
#     Config.MOVING_AVERAGE = 1
#     Config.RUN_TIME = 43200
# elif Config.MODEL_SETTING == 27:
#     # not working
#     Config.DIRECTION = 10
#     Config.REINFORCE = 1
#     Config.STATE_EMBED = 1
#     Config.USE_OR_COST = 1
#     Config.RUN_TIME = 43200
# elif Config.MODEL_SETTING == 28:
#     # repeat
#     Config.DIRECTION = 1
#     Config.FROM_FILE = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     Config.RUN_TIME = 259200
# elif Config.MODEL_SETTING == 29:
#     # repeat
#     Config.DIRECTION = 2
#     Config.FROM_FILE = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     Config.RUN_TIME = 259200
elif Config.MODEL_SETTING == 30:
    # working very well 1.0
    Config.DIRECTION = 3
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RUN_TIME = 259200
# elif Config.MODEL_SETTING == 31:
#     # not working
#     Config.DIRECTION = 4
#     Config.FROM_FILE = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     Config.RUN_TIME = 259200
# elif Config.MODEL_SETTING == 32:
#     # not working
#     Config.DIRECTION = 1
#     Config.FROM_FILE = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     Config.STOCHASTIC = 1
#     Config.RUN_TIME = 259200
# elif Config.MODEL_SETTING == 33:
#     # not working
#     Config.DIRECTION = 4
#     Config.FROM_FILE = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     Config.STOCHASTIC = 1
#     Config.RUN_TIME = 259200
elif Config.MODEL_SETTING == 34:
    # working 1.2 but went back up at end
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.RUN_TIME = 259200
elif Config.MODEL_SETTING == 35:
    # working 1.5 1.3 1.2
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.RNN_HIDDEN_DIM = 32
    # Config.RUN_TIME = 43200
    Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 36:
#     # not working
#     Config.DIRECTION = 10
#     Config.REINFORCE = 1
#     Config.RNN_HIDDEN_DIM = 64
#     Config.RUN_TIME = 43200
elif Config.MODEL_SETTING == 37:
    # working 1.3 1.3
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.MAX_GRAD = 0
    # Config.RUN_TIME = 43200
    Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 38:
#     # not working
#     Config.DIRECTION = 10
#     Config.REINFORCE = 1
#     Config.LEARNING_RATE = 1e-2
#     # Config.RUN_TIME = 43200
#     Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 39:
#     # not working
#     Config.DIRECTION = 10
#     Config.REINFORCE = 1
#     Config.STATE_EMBED = 1
#     Config.RUN_TIME = 259200
# elif Config.MODEL_SETTING == 40:
#     # not working
#     Config.DIRECTION = 10
#     Config.REINFORCE = 1
#     Config.STATE_EMBED = 1
#     Config.MOVING_AVERAGE = 1
#     Config.RUN_TIME = 259200
# elif Config.MODEL_SETTING == 41:
#     # not working
#     Config.DIRECTION = 5
#     Config.FROM_FILE = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     # Config.RUN_TIME = 43200
#     Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 42:
#     # maybe working
#     Config.DIRECTION = 5
#     Config.REINFORCE = 1
#     # Config.RUN_TIME = 43200
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 43:
#     # not working
#     Config.DIRECTION = 5
#     Config.REINFORCE = 1
#     Config.STATE_EMBED = 1
#     # Config.RUN_TIME = 43200
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 44:
#     # not working
#     Config.DIRECTION = 10
#     Config.REINFORCE = 1
#     Config.USE_BAHDANAU = 1
#     Config.RUN_TIME = 43200
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 45:
#     # not working
#     Config.DIRECTION = 10
#     Config.REINFORCE = 1
#     Config.STATE_EMBED = 1
#     Config.USE_BAHDANAU = 1
#     Config.RUN_TIME = 43200
elif Config.MODEL_SETTING == 46:
    # working 1.5 1.1 1.0
    Config.DIRECTION = 10
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.STATE_EMBED = 1
    Config.USE_BAHDANAU = 1
    # Config.RUN_TIME = 43200
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 47:
    # working 1.5 1.1 1.0
    Config.DIRECTION = 10
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.USE_BAHDANAU = 1
    # Config.RUN_TIME = 43200
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 48:
#     # not working
#     Config.DIRECTION = 4
#     Config.FROM_FILE = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     Config.STATE_EMBED = 1
#     Config.USE_BAHDANAU = 1
#     Config.RUN_TIME = 43200
# elif Config.MODEL_SETTING == 49:
#     # maybe working
#     Config.DIRECTION = 4
#     Config.FROM_FILE = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     Config.USE_BAHDANAU = 1
#     # Config.RUN_TIME = 43200
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 50:
    # working 1.4
    Config.DIRECTION = 2
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 259200
elif Config.MODEL_SETTING == 51:
    # working 1.2
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.RNN_HIDDEN_DIM = 16
    # Config.RUN_TIME = 86400
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 52:
    # working 1.4
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.RNN_HIDDEN_DIM = 32
    # Config.RUN_TIME = 86400
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 53:
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.RNN_HIDDEN_DIM = 32
    Config.MAX_GRAD = 0
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 86400
# elif Config.MODEL_SETTING == 54:
#     # not working
#     Config.DIRECTION = 10
#     Config.REINFORCE = 1
#     Config.RNN_HIDDEN_DIM = 32
#     Config.STATE_EMBED = 1
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 86400
# elif Config.MODEL_SETTING == 55:
#     # not working
#     Config.DIRECTION = 10
#     Config.REINFORCE = 1
#     Config.RNN_HIDDEN_DIM = 32
#     Config.STATE_EMBED = 1
#     Config.LR_DECAY_OFF = 1
#     Config.MOVING_AVERAGE = 1
#     Config.RUN_TIME = 86400
elif Config.MODEL_SETTING == 56:
    # working 1.1
    Config.DIRECTION = 5
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RNN_HIDDEN_DIM = 32
    Config.USE_BAHDANAU = 1
    # Config.RUN_TIME = 86400
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 57:
#     # not working
#     Config.DIRECTION = 5
#     Config.REINFORCE = 1
#     Config.RNN_HIDDEN_DIM = 32
#     Config.LR_DECAY_OFF = 1
#     Config.USE_BAHDANAU = 1
#     Config.MOVING_AVERAGE = 1
#     Config.RUN_TIME = 86400
elif Config.MODEL_SETTING == 58:
    # working 1.2
    Config.DIRECTION = 5
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RNN_HIDDEN_DIM = 32
    Config.STATE_EMBED = 1
    # Config.RUN_TIME = 86400
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 59:
#     # not working
#     Config.DIRECTION = 2
#     Config.REINFORCE = 1
#     Config.USE_BAHDANAU = 1
#     Config.MOVING_AVERAGE = 1
#     Config.RUN_TIME = 86400
# elif Config.MODEL_SETTING == 60:
#     # not working
#     Config.DIRECTION = 10
#     Config.FROM_FILE = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     Config.RNN_HIDDEN_DIM = 32
#     Config.LR_DECAY_OFF = 1
#     Config.MAX_GRAD = 0
#     Config.RUN_TIME = 86400
elif Config.MODEL_SETTING == 61:
    # working 1.3 1.3
    Config.DIRECTION = 5
    Config.REINFORCE = 1
    Config.RNN_HIDDEN_DIM = 32
    # Config.RUN_TIME = 86400
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 62:
    # working 1.5
    Config.DIRECTION = 10
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RNN_HIDDEN_DIM = 32
    Config.LR_DECAY_OFF = 1
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 86400
elif Config.MODEL_SETTING == 63:
    # working 1.4
    Config.DIRECTION = 10
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.USE_BAHDANAU = 1
    # Config.RUN_TIME = 86400
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 64:
    # working 1.3
    Config.DIRECTION = 5
    Config.REINFORCE = 1
    Config.USE_BAHDANAU = 1
    Config.RNN_HIDDEN_DIM = 16
    # Config.RUN_TIME = 64800
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 65:
#     # not working
#     Config.DIRECTION = 5
#     Config.REINFORCE = 1
#     Config.USE_BAHDANAU = 1
#     Config.MOVING_AVERAGE = 1
#     Config.LR_DECAY_OFF = 1
#     Config.RNN_HIDDEN_DIM = 16
#     Config.RUN_TIME = 64800
# elif Config.MODEL_SETTING == 66:
#     # not working
#     Config.DIRECTION = 10
#     Config.REINFORCE = 1
#     Config.STATE_EMBED = 1
#     Config.RNN_HIDDEN_DIM = 32
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 64800
# elif Config.MODEL_SETTING == 67:
#     # not working
#     Config.DIRECTION = 6
#     Config.FROM_FILE = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     Config.USE_BAHDANAU = 1
#     Config.RNN_HIDDEN_DIM = 32
#     # Config.RUN_TIME = 64800
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 68:
#     # not working
#     Config.DIRECTION = 6
#     Config.FROM_FILE = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     Config.USE_BAHDANAU = 1
#     Config.RNN_HIDDEN_DIM = 32
#     Config.STATE_EMBED = 1
#     # Config.RUN_TIME = 64800
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 69:
#     # not working
#     Config.DIRECTION = 6
#     Config.FROM_FILE = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     Config.RNN_HIDDEN_DIM = 32
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 64800
# elif Config.MODEL_SETTING == 70:
#     # not working
#     Config.DIRECTION = 6
#     Config.FROM_FILE = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     Config.RNN_HIDDEN_DIM = 32
#     Config.STATE_EMBED = 1
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 64800
# elif Config.MODEL_SETTING == 71:
#     # not working
#     Config.DIRECTION = 6
#     Config.FROM_FILE = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     Config.USE_BAHDANAU = 1
#     Config.RNN_HIDDEN_DIM = 16
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 64800
# elif Config.MODEL_SETTING == 72:
#     # not working
#     Config.DIRECTION = 6
#     Config.FROM_FILE = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     Config.USE_BAHDANAU = 1
#     Config.RNN_HIDDEN_DIM = 16
#     Config.STATE_EMBED = 1
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 64800
# elif Config.MODEL_SETTING == 73:
#     # not working
#     Config.DIRECTION = 6
#     Config.FROM_FILE = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     Config.RNN_HIDDEN_DIM = 16
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 64800
# elif Config.MODEL_SETTING == 74:
#     # not working
#     Config.DIRECTION = 6
#     Config.FROM_FILE = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     Config.RNN_HIDDEN_DIM = 16
#     Config.STATE_EMBED = 1
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 64800
# elif Config.MODEL_SETTING == 75:
#     # repeat?
#     Config.DIRECTION = 10
#     Config.FROM_FILE = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     Config.RNN_HIDDEN_DIM = 32
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 64800
# elif Config.MODEL_SETTING == 76:
#     # not working
#     Config.DIRECTION = 10
#     Config.FROM_FILE = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     Config.RNN_HIDDEN_DIM = 32
#     Config.INPUT_TIME = 1
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 64800
elif Config.MODEL_SETTING == 77:
    # working 1.3
    Config.DIRECTION = 10
    Config.FROM_FILE = 1
    Config.STATE_EMBED = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RNN_HIDDEN_DIM = 32
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 64800
elif Config.MODEL_SETTING == 78:
    # working 1.3
    Config.DIRECTION = 10
    Config.FROM_FILE = 1
    Config.STATE_EMBED = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RNN_HIDDEN_DIM = 32
    Config.INPUT_TIME = 1
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 64800
# elif Config.MODEL_SETTING == 79:
#     # not working
#     Config.DIRECTION = 10
#     Config.FROM_FILE = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     Config.RNN_HIDDEN_DIM = 32
#     Config.INPUT_ALL = 1
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 64800
# elif Config.MODEL_SETTING == 80:
#     # not working
#     # working for the loss but not sampled cost
#     Config.DIRECTION = 5
#     Config.FROM_FILE = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     Config.RNN_HIDDEN_DIM = 32
#     Config.INVERSE_SOFTMAX_TEMP = .8
#     Config.USE_BAHDANAU = 1
#     Config.SAMPLING = 1
#     # Config.RUN_TIME = 64800
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 81:
    # working 1.1
    Config.DIRECTION = 5
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RNN_HIDDEN_DIM = 32
    Config.USE_BAHDANAU = 1
    Config.SAMPLING = 1
    Config.INVERSE_SOFTMAX_TEMP = 10.0
    # Config.RUN_TIME = 64800
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 82:
    # working 1.1
    Config.DIRECTION = 5
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RNN_HIDDEN_DIM = 32
    Config.USE_BAHDANAU = 1
    Config.SAMPLING = 1
    Config.INVERSE_SOFTMAX_TEMP = 100.0
    # Config.RUN_TIME = 64800
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 83:
    # working 1.2
    Config.DIRECTION = 5
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RNN_HIDDEN_DIM = 32
    Config.USE_BAHDANAU = 1
    # Config.RUN_TIME = 64800
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 84:
#     # not working
#     Config.DIRECTION = 5
#     Config.REINFORCE = 1
#     Config.STOCHASTIC = 1
#     Config.INVERSE_SOFTMAX_TEMP = 1.0
#     Config.MOVING_AVERAGE = 1
#     # Config.RUN_TIME = 86400
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 85:
#     # not working
#     Config.DIRECTION = 5
#     Config.REINFORCE = 1
#     Config.STOCHASTIC = 1
#     Config.INVERSE_SOFTMAX_TEMP = 5.0
#     Config.MOVING_AVERAGE = 1
#     # Config.RUN_TIME = 86400
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 86:
#     # not working
#     Config.DIRECTION = 5
#     Config.REINFORCE = 1
#     Config.STOCHASTIC = 1
#     Config.INVERSE_SOFTMAX_TEMP = 10.0
#     Config.RUN_TIME = 86400
# elif Config.MODEL_SETTING == 87:
#     # not working also somehow higher than 10 cost on average
#     Config.DIRECTION = 5
#     Config.REINFORCE = 1
#     Config.STOCHASTIC = 1
#     Config.INVERSE_SOFTMAX_TEMP = .8
#     Config.RUN_TIME = 86400
# elif Config.MODEL_SETTING == 88:
#     # not working
#     Config.DIRECTION = 5
#     Config.REINFORCE = 1
#     Config.STOCHASTIC = 1
#     Config.INVERSE_SOFTMAX_TEMP = 5.0
#     Config.RUN_TIME = 86400
# elif Config.MODEL_SETTING == 89:
#     # not working
#     Config.DIRECTION = 5
#     Config.REINFORCE = 1
#     Config.STOCHASTIC = 1
#     Config.INVERSE_SOFTMAX_TEMP = 10.0
#     Config.RUN_TIME = 86400
# elif Config.MODEL_SETTING == 90:
#     # not working
#     Config.DIRECTION = 2
#     Config.FROM_FILE = 1
#     Config.LOGIT_CLIP_SCALAR = 10
#     Config.USE_BAHDANAU = 1
#     Config.SAME_BATCH = 1
#     Config.RUN_TIME = 259200
# elif Config.MODEL_SETTING == 91:
#     # maybe working but prob not
#     Config.DIRECTION = 10
#     Config.REINFORCE = 1
#     Config.TRAINING_MIN_BATCH_SIZE = 64
#     Config.USE_PPO = 1
#     Config.TYPE_1 = 1
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 86400
# elif Config.MODEL_SETTING == 92:
#     # not working
#     Config.DIRECTION = 10
#     Config.REINFORCE = 1
#     Config.LEARNING_RATE = 1e-2
#     Config.TRAINING_MIN_BATCH_SIZE = 64
#     Config.USE_PPO = 1
#     # Config.RUN_TIME = 86400
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 93:
#     # not working
#     Config.DIRECTION = 10
#     Config.REINFORCE = 1
#     Config.TRAINING_MIN_BATCH_SIZE = 64
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 86400
# elif Config.MODEL_SETTING == 94:
#     # not working
#     Config.DIRECTION = 10
#     Config.REINFORCE = 1
#     Config.TRAINING_MIN_BATCH_SIZE = 64
#     Config.USE_BAHDANAU = 1
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 86400
# elif Config.MODEL_SETTING == 95:
#     # not working
#     Config.DIRECTION = 2
#     Config.REINFORCE = 1
#     Config.SEQUENCE_COST = 1
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 86400
# elif Config.MODEL_SETTING == 96:
#     # not working
#     Config.DIRECTION = 2
#     Config.REINFORCE = 1
#     Config.USE_BAHDANAU = 1
#     Config.SEQUENCE_COST = 1
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 86400
# elif Config.MODEL_SETTING == 97:
#     # not working
#     Config.DIRECTION = 2
#     Config.REINFORCE = 1
#     Config.SEQUENCE_COST = 1
#     Config.USE_PPO = 1
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 86400
# elif Config.MODEL_SETTING == 98:
#     # not working
#     Config.DIRECTION = 2
#     Config.REINFORCE = 1
#     Config.USE_BAHDANAU = 1
#     Config.SEQUENCE_COST = 1
#     Config.USE_PPO = 1
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 86400
# elif Config.MODEL_SETTING == 99:
#     # not working
#     Config.DIRECTION = 5
#     Config.REINFORCE = 1
#     Config.USE_BAHDANAU = 1
#     Config.SEQUENCE_COST = 1
#     Config.USE_PCA = 1
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 86400
# elif Config.MODEL_SETTING == 100:
#     # not working
#     Config.DIRECTION = 5
#     Config.REINFORCE = 1
#     Config.SEQUENCE_COST = 1
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 86400
# elif Config.MODEL_SETTING == 101:
#     # not working
#     Config.DIRECTION = 5
#     Config.REINFORCE = 1
#     Config.SEQUENCE_COST = 1
#     Config.USE_PCA = 1
#     # Config.RUN_TIME = 86400
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 102:
#     # not working
#     Config.DIRECTION = 5
#     Config.REINFORCE = 1
#     Config.USE_PCA = 1
#     # Config.RUN_TIME = 86400
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 103:
#     # not working
#     Config.DIRECTION = 5
#     # Config.SAME_BATCH = 1
#     # Config.USE_PPO = 1
#     # Config.NUM_PPO_EPOCH = 5
#     Config.SEQUENCE_COST = 1
#     Config.STOCHASTIC = 1
#     Config.REINFORCE = 1
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 86400
# elif Config.MODEL_SETTING == 104:
#     # not working
#     Config.DIRECTION = 5
#     Config.REINFORCE = 1
#     Config.USE_PPO = 1
#     Config.LR_DECAY_OFF = 1
#     Config.RUN_TIME = 86400
# elif Config.MODEL_SETTING == 105:
#     # not working
#     Config.DIRECTION = 5
#     Config.REINFORCE = 1
#     Config.LR_DECAY_OFF = 1
#     Config.STOCHASTIC = 1
#     Config.INVERSE_SOFTMAX_TEMP = 5.0
#     Config.SEQUENCE_COST = 1
#     Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 106:
#     # not working
#     Config.DIRECTION = 5
#     Config.REINFORCE = 1
#     Config.LR_DECAY_OFF = 1
#     Config.STOCHASTIC = 1
#     Config.INVERSE_SOFTMAX_TEMP = 1.0
#     Config.SEQUENCE_COST = 1
#     Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 107:
#     # not working
#     Config.DIRECTION = 5
#     Config.REINFORCE = 1
#     Config.LR_DECAY_OFF = 1
#     Config.STOCHASTIC = 1
#     Config.INVERSE_SOFTMAX_TEMP = 10.0
#     Config.SEQUENCE_COST = 1
#     Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 108:
#     # not working
#     Config.DIRECTION = 5
#     Config.REINFORCE = 1
#     Config.LR_DECAY_OFF = 1
#     Config.STOCHASTIC = 1
#     Config.INVERSE_SOFTMAX_TEMP = 5.0
#     Config.SEQUENCE_COST = 1
#     Config.TRAINING_MIN_BATCH_SIZE = 64
#     Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 109:
#     # not working
#     Config.DIRECTION = 5
#     Config.REINFORCE = 1
#     Config.LR_DECAY_OFF = 1
#     Config.STOCHASTIC = 1
#     Config.INVERSE_SOFTMAX_TEMP = 1.0
#     Config.SEQUENCE_COST = 1
#     Config.TRAINING_MIN_BATCH_SIZE = 64
#     Config.RUN_TIME = 129600
# elif Config.MODEL_SETTING == 110:
#     # not working
#     Config.DIRECTION = 5
#     Config.REINFORCE = 1
#     Config.LR_DECAY_OFF = 1
#     Config.STOCHASTIC = 1
#     Config.INVERSE_SOFTMAX_TEMP = 10.0
#     Config.SEQUENCE_COST = 1
#     Config.TRAINING_MIN_BATCH_SIZE = 64
#     Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 200:
    Config.DIRECTION = 1
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-2
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 201:
    Config.DIRECTION = 1
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 202:
    Config.DIRECTION = 1
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-4
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 203:
    Config.DIRECTION = 1
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 32
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 204:  # optimal 1.002
    Config.DIRECTION = 1
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 64
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 205:  # optimal 1.001
    Config.DIRECTION = 1
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 128
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 206:
    Config.DIRECTION = 1
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-2
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 207:
    Config.DIRECTION = 1
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 208:
    Config.DIRECTION = 1
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-4
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 209:
    Config.DIRECTION = 1
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 32
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 210:
    Config.DIRECTION = 1
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 64
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 211:
    Config.DIRECTION = 1
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 128
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 212:
    Config.DIRECTION = 5
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-2
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 213:
    Config.DIRECTION = 5
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 214:
    Config.DIRECTION = 5
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-4
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 215:
    Config.DIRECTION = 5
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 32
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 216:
    Config.DIRECTION = 5
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 64
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 217:
    Config.DIRECTION = 5
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 128
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 218:  # optimal 1.232
    Config.DIRECTION = 5
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-2
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 219:  # optimal 1.059
    Config.DIRECTION = 5
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 220:  # optimal 1.053
    Config.DIRECTION = 5
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-4
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 221:  # optimal 1.119
    Config.DIRECTION = 5
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 32
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 222:  # optimal 1.073
    Config.DIRECTION = 5
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 64
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 223:  # optimal 1.055
    Config.DIRECTION = 5
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 128
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 224:
    Config.DIRECTION = 2
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-2
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 225:  # optimal 1.128
    Config.DIRECTION = 2
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 226:
    Config.DIRECTION = 2
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-4
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 227:
    Config.DIRECTION = 2
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 32
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 228:
    Config.DIRECTION = 2
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 64
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 229:
    Config.DIRECTION = 2
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 128
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 230:
    Config.DIRECTION = 2
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-2
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 231:
    Config.DIRECTION = 2
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 232:
    Config.DIRECTION = 2
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-4
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 233:  # optimal 1.0001
    Config.DIRECTION = 2
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 32
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 234:
    Config.DIRECTION = 2
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 64
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 235:
    Config.DIRECTION = 2
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 128
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 300:
    Config.DIRECTION = 1
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-2
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 301:
    Config.DIRECTION = 1
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 302:
    Config.DIRECTION = 1
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-4
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 303:
    Config.DIRECTION = 1
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 32
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 304:  # optimal 1.002
    Config.DIRECTION = 1
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 64
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 305:  # optimal 1.001
    Config.DIRECTION = 1
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 128
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 306:
    Config.DIRECTION = 1
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-2
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 307:
    Config.DIRECTION = 1
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 308:
    Config.DIRECTION = 1
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-4
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 309:
    Config.DIRECTION = 1
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 32
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 310:
    Config.DIRECTION = 1
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 64
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 311:
    Config.DIRECTION = 1
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 128
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 312:
    Config.DIRECTION = 5
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-2
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 313:
    Config.DIRECTION = 5
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 314:
    Config.DIRECTION = 5
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-4
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 315:
    Config.DIRECTION = 5
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 32
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 316:
    Config.DIRECTION = 5
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 64
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 317:
    Config.DIRECTION = 5
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 128
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 318:
    Config.DIRECTION = 5
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-2
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 319:
    Config.DIRECTION = 5
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 320:
    Config.DIRECTION = 5
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-4
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 321:
    Config.DIRECTION = 5
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 32
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 322:
    Config.DIRECTION = 5
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 64
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 323:
    Config.DIRECTION = 5
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 128
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 324:
    Config.DIRECTION = 2
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-2
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 325:
    Config.DIRECTION = 2
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 326:
    Config.DIRECTION = 2
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-4
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 327:
    Config.DIRECTION = 2
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 32
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 328:
    Config.DIRECTION = 2
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 64
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 329:
    Config.DIRECTION = 2
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 128
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 330:
    Config.DIRECTION = 2
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-2
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 331:
    Config.DIRECTION = 2
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 332:
    Config.DIRECTION = 2
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-4
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 333:
    Config.DIRECTION = 2
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 32
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 334:
    Config.DIRECTION = 2
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 64
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
elif Config.MODEL_SETTING == 335:
    Config.DIRECTION = 2
    Config.REINFORCE = 1
    Config.LR_DECAY_OFF = 1
    Config.LEARNING_RATE = 1e-3
    Config.RNN_HIDDEN_DIM = 128
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 129600
else:
    sys.exit()

ConfigParse = configparser.ConfigParser()
ConfigParse.optionxform = str
variables = [attr for attr in dir(Config) if not callable(getattr(Config, attr)) and not attr.startswith("__")]
if Config.RESTORE == 1:
    print()
    print("RESTORING...")
    ConfigParse.read(Config.PATH + "ini_files/" + Config.MODEL_NAME + ".ini")
    print(ConfigParse.sections())
    for var in variables:
        print(var, type(getattr(Config, var))(ConfigParse.get("VARIABLES", var)))
        setattr(Config, var, type(getattr(Config, var))(ConfigParse.get("VARIABLES", var)))
    setattr(Config, "RESTORE", 1)
    setattr(Config, "MODEL_NAME", Config.MODEL_NAME)
else:
    cfgfile = open(Config.PATH + "ini_files/" + Config.MODEL_NAME + ".ini", 'w')
    ConfigParse.add_section('VARIABLES')
    print()
    for var in variables:
        print(var, str(getattr(Config, var)))
        ConfigParse.set('VARIABLES', var, str(getattr(Config, var)))
    ConfigParse.write(cfgfile)
    cfgfile.close()

print()
Server().main()
