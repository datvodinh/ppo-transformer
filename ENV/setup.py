#  SHOT_PATH = 'A:/AutoTrain/GAME/'
DRIVE_FOLDER = "G:/My Drive/AutomaticColab/"


SHORT_PATH = ""
#  DRIVE_FOLDER = 'H:/Drive của tôi/AutomaticColab/'

import importlib.util
import sys


def make(game_name):
    def add_game_to_syspath(game_name):
        if len(sys.argv) >= 2:
            sys.argv = [sys.argv[0]]
        sys.argv.append(game_name)

    def setup_game(game_name):
        spec = importlib.util.spec_from_file_location(
            "env", f"{SHORT_PATH}Base/{game_name}/env.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    add_game_to_syspath(game_name)
    env = setup_game(game_name)

    return env


def setup_game(game_name):
    spec = importlib.util.spec_from_file_location(
        "env", f"{SHORT_PATH}Base/{game_name}/env.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_module_player(player, game_name=None):
    if game_name == None:
        spec = importlib.util.spec_from_file_location(
            "Agent_player", f"{SHORT_PATH}Agent/{player}/Agent_player.py"
        )
    else:
        spec = importlib.util.spec_from_file_location(
            "Agent_player", f"{SHORT_PATH}Agent/Ifelse/{game_name}/{player}.py"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
