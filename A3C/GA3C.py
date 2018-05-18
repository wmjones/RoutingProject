import sys
import configparser
from Config import Config
from Server import Server

for i in range(1, len(sys.argv)):
    x, y = sys.argv[i].split('=')
    setattr(Config, x, type(getattr(Config, x))(y))

ConfigParse = configparser.ConfigParser()
ConfigParse.optionxform = str
variables = [attr for attr in dir(Config) if not callable(getattr(Config, attr)) and not attr.startswith("__")]
if Config.MODEL_SETTING == 1:
    Config.DIRECTION = 2
    Config.REINFORCE = 1
    Config.TRAINING_MIN_BATCH_SIZE = 30
    Config.LOGIT_CLIP_SCALAR = 10
elif Config.MODEL_SETTING == 2:
    Config.DIRECTION = 2
    Config.REINFORCE = 1
    Config.MOVING_AVERAGE = 1
    Config.TRAINING_MIN_BATCH_SIZE = 30
elif Config.MODEL_SETTING == 3:
    Config.DIRECTION = 2
    Config.REINFORCE = 1
    Config.TRAINING_MIN_BATCH_SIZE = 30
    Config.USE_OR_COST = 1
# elif Config.MODEL_SETTING == 4:
#     Config.DIRECTION = 10
#     Config.REINFORCE = 1
# elif Config.MODEL_SETTING == 5:
#     Config.DIRECTION = 2
#     Config.REINFORCE = 1
# elif Config.MODEL_SETTING == 6:
#     Config.DIRECTION = 2
#     Config.REINFORCE = 1
#     Config.MOVING_AVERAGE = 1
# elif Config.MODEL_SETTING == 7:
#     Config.DIRECTION = 1
#     Config.STATE_EMBED = 1
#     Config.REINFORCE = 1
# elif Config.MODEL_SETTING == 8:
#     Config.DIRECTION = 3
#     Config.STATE_EMBED = 1
#     Config.REINFORCE = 1
# elif Config.MODEL_SETTING == 9:
#     Config.DIRECTION = 8
#     Config.STOCHASTIC = 1
#     Config.FROM_FILE = 0
#     Config.STATE_EMBED = 0
# elif Config.MODEL_SETTING == 10:
#     Config.DIRECTION = 8
#     Config.STATE_EMBED = 1
#     Config.REINFORCE = 1
# elif Config.MODEL_SETTING == 11:
#     Config.DIRECTION = 1
#     Config.REINFORCE = 1
#     Config.MOVING_AVERAGE = 1
# elif Config.MODEL_SETTING == 12:
#     Config.DIRECTION = 3
#     Config.STATE_EMBED = 1
#     Config.REINFORCE = 1
#     Config.MOVING_AVERAGE = 1
elif Config.MODEL_SETTING == 13:
    Config.DIRECTION = 9
    Config.REINFORCE = 1
# elif Config.MODEL_SETTING == 14:
#     Config.DIRECTION = 9
#     Config.REINFORCE = 1
#     Config.REZA = 1
if Config.REINFORCE == 0 and Config.LOGIT_CLIP_SCALAR == 0:
    print()
    print("You need to set LOGIT_CLIP_SCALAR to not be zero!")
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
