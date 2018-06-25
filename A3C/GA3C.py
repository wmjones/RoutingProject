import sys
import configparser
from Config import Config
from Server import Server

for i in range(1, len(sys.argv)):
    x, y = sys.argv[i].split('=')
    setattr(Config, x, type(getattr(Config, x))(y))

if Config.MODEL_SETTING == 1:
    Config.DIRECTION = 1
    Config.USE_BAHDANAU = 1
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 64800
if Config.MODEL_SETTING == 2:
    Config.DIRECTION = 1
    Config.USE_BAHDANAU = 1
    Config.REINFORCE = 1
    Config.RNN_HIDDEN_DIM = 16
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 64800
if Config.MODEL_SETTING == 3:
    Config.DIRECTION = 2
    Config.USE_BAHDANAU = 1
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 64800
if Config.MODEL_SETTING == 4:
    Config.DIRECTION = 2
    Config.USE_BAHDANAU = 1
    Config.REINFORCE = 1
    Config.RNN_HIDDEN_DIM = 16
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 64800
if Config.MODEL_SETTING == 5:
    Config.DIRECTION = 3
    Config.USE_BAHDANAU = 1
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RUN_TIME = 64800
if Config.MODEL_SETTING == 6:
    Config.DIRECTION = 3
    Config.USE_BAHDANAU = 1
    Config.REINFORCE = 1
    Config.RUN_TIME = 64800
if Config.MODEL_SETTING == 7:
    Config.DIRECTION = 4
    Config.USE_BAHDANAU = 1
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RUN_TIME = 64800
if Config.MODEL_SETTING == 8:
    Config.DIRECTION = 4
    Config.USE_BAHDANAU = 1
    Config.REINFORCE = 1
    Config.RUN_TIME = 64800
if Config.MODEL_SETTING == 9:
    Config.DIRECTION = 10
    Config.STATE_EMBED = 1
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 10:
    Config.DIRECTION = 10
    Config.STATE_EMBED = 1
    Config.REINFORCE = 1
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 11:
    Config.DIRECTION = 10
    Config.STATE_EMBED = 1
    Config.REINFORCE = 1
    Config.TRAINING_MIN_BATCH_SIZE = 30
    Config.USE_OR_COST = 1
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 12:
    Config.DIRECTION = 1
    Config.STATE_EMBED = 1
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 13:
    Config.DIRECTION = 1
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.STOCHASTIC = 1
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 14:
    Config.DIRECTION = 1
    Config.REINFORCE = 1
    Config.USE_OR_COST = 1
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 15:
    Config.DIRECTION = 2
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.STOCHASTIC = 1
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 16:
    Config.DIRECTION = 2
    Config.REINFORCE = 1
    Config.USE_OR_COST = 1
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 17:
    Config.DIRECTION = 3
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.STOCHASTIC = 1
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 18:
    Config.DIRECTION = 3
    Config.REINFORCE = 1
    Config.USE_OR_COST = 1
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 19:
    Config.DIRECTION = 4
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.STOCHASTIC = 1
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 20:
    Config.DIRECTION = 4
    Config.REINFORCE = 1
    Config.USE_OR_COST = 1
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 21:
    Config.DIRECTION = 10
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 22:
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 23:
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.MOVING_AVERAGE = 1
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 24:
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.USE_OR_COST = 1
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 25:
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.STATE_EMBED = 1
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 26:
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.STATE_EMBED = 1
    Config.MOVING_AVERAGE = 1
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 27:
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.STATE_EMBED = 1
    Config.USE_OR_COST = 1
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 28:
    Config.DIRECTION = 1
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RUN_TIME = 259200
if Config.MODEL_SETTING == 29:
    Config.DIRECTION = 2
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RUN_TIME = 259200
if Config.MODEL_SETTING == 30:
    Config.DIRECTION = 3
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RUN_TIME = 259200
if Config.MODEL_SETTING == 31:
    Config.DIRECTION = 4
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RUN_TIME = 259200
if Config.MODEL_SETTING == 32:
    Config.DIRECTION = 1
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.STOCHASTIC = 1
    Config.RUN_TIME = 259200
if Config.MODEL_SETTING == 33:
    Config.DIRECTION = 4
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.STOCHASTIC = 1
    Config.RUN_TIME = 259200
if Config.MODEL_SETTING == 34:
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.RUN_TIME = 259200
if Config.MODEL_SETTING == 35:
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.RNN_HIDDEN_DIM = 32
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 36:
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.RNN_HIDDEN_DIM = 64
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 37:
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.MAX_GRAD = 0
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 38:
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.LEARNING_RATE = 1e-2
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 39:
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.STATE_EMBED = 1
    Config.RUN_TIME = 259200
if Config.MODEL_SETTING == 40:
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.STATE_EMBED = 1
    Config.MOVING_AVERAGE = 1
    Config.RUN_TIME = 259200
if Config.MODEL_SETTING == 41:
    Config.DIRECTION = 5
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 42:
    Config.DIRECTION = 5
    Config.REINFORCE = 1
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 43:
    Config.DIRECTION = 5
    Config.REINFORCE = 1
    Config.STATE_EMBED = 1
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 44:
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 45:
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.STATE_EMBED = 1
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 46:
    Config.DIRECTION = 10
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.STATE_EMBED = 1
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 47:
    Config.DIRECTION = 10
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 48:
    Config.DIRECTION = 4
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.STATE_EMBED = 1
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 49:
    Config.DIRECTION = 4
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 43200
if Config.MODEL_SETTING == 50:
    Config.DIRECTION = 2
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 259200
if Config.MODEL_SETTING == 51:
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.RNN_HIDDEN_DIM = 16
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 86400
if Config.MODEL_SETTING == 52:
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.RNN_HIDDEN_DIM = 32
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 86400
if Config.MODEL_SETTING == 53:
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.RNN_HIDDEN_DIM = 32
    Config.MAX_GRAD = 0
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 86400
if Config.MODEL_SETTING == 54:
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.RNN_HIDDEN_DIM = 32
    Config.STATE_EMBED = 1
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 86400
if Config.MODEL_SETTING == 55:
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.RNN_HIDDEN_DIM = 32
    Config.STATE_EMBED = 1
    Config.LR_DECAY_OFF = 1
    Config.MOVING_AVERAGE = 1
    Config.RUN_TIME = 86400
if Config.MODEL_SETTING == 56:
    Config.DIRECTION = 5
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RNN_HIDDEN_DIM = 32
    Config.USE_BAHDANAU = 1
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 86400
if Config.MODEL_SETTING == 57:
    Config.DIRECTION = 5
    Config.REINFORCE = 1
    Config.RNN_HIDDEN_DIM = 32
    Config.LR_DECAY_OFF = 1
    Config.USE_BAHDANAU = 1
    Config.MOVING_AVERAGE = 1
    Config.RUN_TIME = 86400
if Config.MODEL_SETTING == 58:
    Config.DIRECTION = 5
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RNN_HIDDEN_DIM = 32
    Config.LR_DECAY_OFF = 1
    Config.STATE_EMBED = 1
    Config.RUN_TIME = 86400
if Config.MODEL_SETTING == 59:
    Config.DIRECTION = 2
    Config.REINFORCE = 1
    Config.USE_BAHDANAU = 1
    Config.MOVING_AVERAGE = 1
    Config.RUN_TIME = 86400
if Config.MODEL_SETTING == 60:
    Config.DIRECTION = 10
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RNN_HIDDEN_DIM = 32
    Config.LR_DECAY_OFF = 1
    Config.MAX_GRAD = 0
    Config.RUN_TIME = 86400
if Config.MODEL_SETTING == 61:
    Config.DIRECTION = 5
    Config.REINFORCE = 1
    Config.RNN_HIDDEN_DIM = 32
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 86400
if Config.MODEL_SETTING == 62:
    Config.DIRECTION = 10
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RNN_HIDDEN_DIM = 32
    Config.LR_DECAY_OFF = 1
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 86400
if Config.MODEL_SETTING == 63:
    Config.DIRECTION = 10
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.LR_DECAY_OFF = 1
    Config.USE_BAHDANAU = 1
    Config.RUN_TIME = 86400
if Config.MODEL_SETTING == 64:
    Config.DIRECTION = 5
    Config.REINFORCE = 1
    Config.USE_BAHDANAU = 1
    Config.LR_DECAY_OFF = 1
    Config.RNN_HIDDEN_DIM = 16
    Config.RUN_TIME = 64800
if Config.MODEL_SETTING == 65:
    Config.DIRECTION = 5
    Config.REINFORCE = 1
    Config.USE_BAHDANAU = 1
    Config.MOVING_AVERAGE = 1
    Config.LR_DECAY_OFF = 1
    Config.RNN_HIDDEN_DIM = 16
    Config.RUN_TIME = 64800
if Config.MODEL_SETTING == 66:
    Config.DIRECTION = 10
    Config.STATE_EMBED = 1
    Config.REINFORCE = 1
    Config.RNN_HIDDEN_DIM = 32
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 64800
if Config.MODEL_SETTING == 67:
    Config.DIRECTION = 6
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.USE_BAHDANAU = 1
    Config.RNN_HIDDEN_DIM = 32
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 64800
if Config.MODEL_SETTING == 68:
    Config.DIRECTION = 6
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.USE_BAHDANAU = 1
    Config.RNN_HIDDEN_DIM = 32
    Config.STATE_EMBED = 1
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 64800
if Config.MODEL_SETTING == 69:
    Config.DIRECTION = 6
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RNN_HIDDEN_DIM = 32
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 64800
if Config.MODEL_SETTING == 70:
    Config.DIRECTION = 6
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RNN_HIDDEN_DIM = 32
    Config.STATE_EMBED = 1
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 64800
if Config.MODEL_SETTING == 71:
    Config.DIRECTION = 6
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.USE_BAHDANAU = 1
    Config.RNN_HIDDEN_DIM = 16
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 64800
if Config.MODEL_SETTING == 72:
    Config.DIRECTION = 6
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.USE_BAHDANAU = 1
    Config.RNN_HIDDEN_DIM = 16
    Config.STATE_EMBED = 1
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 64800
if Config.MODEL_SETTING == 73:
    Config.DIRECTION = 6
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RNN_HIDDEN_DIM = 16
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 64800
if Config.MODEL_SETTING == 74:
    Config.DIRECTION = 6
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RNN_HIDDEN_DIM = 16
    Config.STATE_EMBED = 1
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 64800
if Config.MODEL_SETTING == 75:
    Config.DIRECTION = 10
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RNN_HIDDEN_DIM = 32
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 64800
if Config.MODEL_SETTING == 76:
    Config.DIRECTION = 10
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RNN_HIDDEN_DIM = 32
    Config.INPUT_TIME = 1
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 64800
if Config.MODEL_SETTING == 77:
    Config.DIRECTION = 10
    Config.FROM_FILE = 1
    Config.STATE_EMBED = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RNN_HIDDEN_DIM = 32
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 64800
if Config.MODEL_SETTING == 78:
    Config.DIRECTION = 10
    Config.FROM_FILE = 1
    Config.STATE_EMBED = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RNN_HIDDEN_DIM = 32
    Config.INPUT_TIME = 1
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 64800
if Config.MODEL_SETTING == 79:
    Config.DIRECTION = 10
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RNN_HIDDEN_DIM = 32
    Config.INPUT_ALL = 1
    Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 64800
if Config.MODEL_SETTING == 80:
    Config.DIRECTION = 5
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RNN_HIDDEN_DIM = 32
    # Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 64800
    Config.USE_BAHDANAU = 1
    Config.SAMPLING = 1
    Config.INVERSE_SOFTMAX_TEMP = .8
if Config.MODEL_SETTING == 81:
    Config.DIRECTION = 5
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RNN_HIDDEN_DIM = 32
    # Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 64800
    Config.USE_BAHDANAU = 1
    Config.SAMPLING = 1
    Config.INVERSE_SOFTMAX_TEMP = 10.0
if Config.MODEL_SETTING == 82:
    Config.DIRECTION = 5
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RNN_HIDDEN_DIM = 32
    # Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 64800
    Config.USE_BAHDANAU = 1
    Config.SAMPLING = 1
    Config.INVERSE_SOFTMAX_TEMP = 100.0
if Config.MODEL_SETTING == 83:
    Config.DIRECTION = 5
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.RNN_HIDDEN_DIM = 32
    # Config.LR_DECAY_OFF = 1
    Config.RUN_TIME = 64800
    Config.USE_BAHDANAU = 1
if Config.MODEL_SETTING == 84:
    Config.DIRECTION = 5
    Config.REINFORCE = 1
    Config.STOCHASTIC = 1
    Config.INVERSE_SOFTMAX_TEMP = 1.0
    Config.RUN_TIME = 86400
    Config.MOVING_AVERAGE = 1
if Config.MODEL_SETTING == 85:
    Config.DIRECTION = 5
    Config.REINFORCE = 1
    Config.STOCHASTIC = 1
    Config.INVERSE_SOFTMAX_TEMP = 5.0
    Config.RUN_TIME = 86400
    Config.MOVING_AVERAGE = 1
if Config.MODEL_SETTING == 86:
    Config.DIRECTION = 5
    Config.REINFORCE = 1
    Config.STOCHASTIC = 1
    Config.INVERSE_SOFTMAX_TEMP = 10.0
    Config.RUN_TIME = 86400
if Config.MODEL_SETTING == 87:
    Config.DIRECTION = 5
    Config.REINFORCE = 1
    Config.STOCHASTIC = 1
    Config.INVERSE_SOFTMAX_TEMP = .8
    Config.RUN_TIME = 86400
if Config.MODEL_SETTING == 88:
    Config.DIRECTION = 5
    Config.REINFORCE = 1
    Config.STOCHASTIC = 1
    Config.INVERSE_SOFTMAX_TEMP = 5.0
    Config.RUN_TIME = 86400
if Config.MODEL_SETTING == 89:
    Config.DIRECTION = 5
    Config.REINFORCE = 1
    Config.STOCHASTIC = 1
    Config.INVERSE_SOFTMAX_TEMP = 10.0
    Config.RUN_TIME = 86400
if Config.MODEL_SETTING == 90:
    Config.DIRECTION = 2
    Config.FROM_FILE = 1
    Config.LOGIT_CLIP_SCALAR = 10
    Config.USE_BAHDANAU = 1
    Config.SAME_BATCH = 1
    Config.RUN_TIME = 259200
if Config.MODEL_SETTING == 91:
    Config.DIRECTION = 10
    Config.REINFORCE = 1
    Config.RUN_TIME = 86400
    Config.SEQUENCE_COST = 1


# if Config.MODEL_SETTING ==
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
