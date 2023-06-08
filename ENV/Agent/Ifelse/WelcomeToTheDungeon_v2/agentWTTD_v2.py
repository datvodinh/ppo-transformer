import sys
import numpy as np
from numba import njit
from numba.typed import List
from setup import setup_game

game_name = sys.argv[1]
env = setup_game(game_name)


def DataAgent():
    per = []
    per.append(np.zeros(1))
    return per


def Test(state, per):
    validActions = env.getValidActions(state)
    actions = np.where(validActions)[0]

    if state[13] > 7:
        if 0 in actions:
            return 0, per

    if 11 in actions:
        return 11, per

    if state[53]:
        if np.sum(state[76:83]) and 1 in actions:
            return 1, per

        for i in np.array([3, 4, 6, 7, 5, 2]):
            if i in actions:
                return i, per

    if state[61]:
        if np.sum(state[74:81]) and 1 in actions:
            return 1, per

        for i in np.array([6, 2, 3, 5, 7, 4]):
            if i in actions:
                return i, per

    act = np.random.choice(actions)
    return act, per
