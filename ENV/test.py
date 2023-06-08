import numpy as np
from numba import njit

from setup import make
from tests.CheckEnv import check_env

env = make("RockPaperScissors")
print(check_env(env))


@njit()
def Test(state, perData):
    validActions = env.getValidActions(state)
    arr_action = np.where(validActions == 1)[0]
    idx = np.random.randint(0, arr_action.shape[0])
    return arr_action[idx], perData


# win1, per = numba_main_2(Test, 1000, np.array([0]), 0)
# print(win1)
# a = time.process_time()
# win2, per = numba_main_2(Test, 10000, np.array([0]), 1)
# b = time.process_time()
# print(win2, b - a)

# a = time.process_time()
# win3, per = numba_main_2(Test, 10000, np.array([0]), -1)
# b = time.process_time()
# print(win3, b - a)
