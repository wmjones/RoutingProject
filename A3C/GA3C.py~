import sys

from Config import Config
from Server import Server

for i in range(1, len(sys.argv)):
    x, y = sys.argv[i].split('=')
    setattr(Config, x, type(getattr(Config, x))(y))  # no idea what this is

# if Config.PLAY_MODE:
#     Config.AGENTS = 1
#     Config.PREDICTORS = 1
#     Config.TRAINERS = 1

Server().main()
