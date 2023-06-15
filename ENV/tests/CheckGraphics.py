import warnings

# from setup import make

warnings.filterwarnings("ignore")
from numba.core.errors import (
    NumbaDeprecationWarning,
    NumbaExperimentalFeatureWarning,
    NumbaPendingDeprecationWarning,
    NumbaWarning,
)

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaExperimentalFeatureWarning)
warnings.simplefilter("ignore", category=NumbaWarning)

import os


def check_graphics(game_name):
    BOOL_CHECK_GRAPHICS = True
    msg = []
    if os.path.exists(f"src/Base/{game_name}/_render_func.py") == False:
        BOOL_CHECK_GRAPHICS = False
        msg.append("_render_func.py not found")
    elif os.path.exists(f"src/Base/{game_name}/env.py") == False:
        BOOL_CHECK_GRAPHICS = False
        msg.append("env.py not found")
    # else:
    # try:
    # env = make(game_name)
    # env.render(Agent=env.bot_lv0, per_data=[0], level=0, max_temp_frame=100)
    # except:
    #     BOOL_CHECK_GRAPHICS = False
    #     msg.append("render() not found")
    return BOOL_CHECK_GRAPHICS, msg
