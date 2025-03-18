import os.path

import matplotlib.pyplot as plt
from Env_SingleBat import sinBATenv

from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy, CnnPolicy
import wandb
import random
import argparse
from datetime import datetime
from collections import deque
from Utils import *
from Exp_Pyctrl_solartron.Exp_ctrl import pycontrol_single
from Parameters import General_param
Gen = General_param


def random_cycling():
    print('====== random action ======')
    action_random = random.sample(range(Gen.action_number),1)  # 得到的是list类型
    action_num = action_random[0]

    exppath, datapath = pathload(action_num)
    times, currents, voltages = pycontrol_single(exppath, datapath, Virtual=False)  # current -> mA

    data = inst_cycles_process(times, currents, voltages)

    s_array = data2shape_array(data)

    # reward record
    StagePred = stagemodel(s_array)

    vel_scap_list = velocity_detect(data)

    reward = 0

    step_count = Gen.current_steps
    save_time = Gen.TIME_NOW
    data_monitor_save(data, action_num, reward, StagePred, vel_scap_list, step_count, save_time)

    crit_Stage_decay.append(StagePred)
    state = [s_array]

    Gen.current_steps += 1

    return np.array(state)


if __name__ == '__main__':

    test_step_num = 1000

    parser = argparse.ArgumentParser()
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-Re_time', type=str, default=None, help='time for data saving files')
    args = parser.parse_args()

    if args.resume:
        Gen.TIME_NOW = args.Re_time
        try:
            monitor_txt = 'post_process\\monitor_save-' + args.Re_time + '/cycling data.txt'
            with open(monitor_txt, "r") as file:
                done_steps = int(file.readlines()[-1].split("\t")[0])
            # 恢复并更新步数和回合数
            Gen.current_steps = done_steps + 1
        except FileNotFoundError:
            print("No monitor_txt file, please check\n")
    else:
        DATE_FORMAT = Gen.DATE_FORMAT
        # time when we run the script
        Gen.TIME_NOW = datetime.now().strftime(DATE_FORMAT)
    print('random charging testing...')

    crit_Stage_decay = deque([0 for _ in range(10)], maxlen=10)
    done_Stage_decay = all(_ == Gen.limit_Stage for _ in crit_Stage_decay)

    '''random cycling'''
    for _ in range(test_step_num):
        random_cycling()
        if done_Stage_decay:
            print('=================================')
            print(" || Please replace a new battery. ||")
            print('=================================')
            sys.exit()

    print('battery test finished')

