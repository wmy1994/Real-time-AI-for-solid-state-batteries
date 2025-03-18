import gym
from gym import error, spaces, utils, logger
from torch.utils.tensorboard import SummaryWriter

import random
import numpy as np
from collections import deque

from Exp_Pyctrl_solartron.Exp_ctrl_double import pycontrol_single, pycontrol_double
from Utils import *

from Parameters import *
Gen = General_param
Inst_p = Instrument_param


class sinBATenv(gym.Env):

    def __init__(self, log_state=True):
        self.save_time = Gen.TIME_NOW

        self.cap = Gen.capacity  # Amp*seconds
        # self.dt = Gen.dt
        # self.C_rate_basic = Gen.C_rate_basic
        # self.ifRGB = Gen.ifRGB

        self.I_unit = Gen.I_unit

        # self.limit_SOH = Gen.limit_SOH
        # self.crit_SOH = deque([1, 1, 1], maxlen=3)

        # self.ifInflection = Gen.ifInflection
        # self.ifStage = Gen.ifStage
        self.limit_Stage = Gen.limit_Stage
        self.crit_Stage_decay = deque([0 for _ in range(500)], maxlen=500)  # 监测50次连续的Stage

        self.training_duration = Gen.training_duration

        self.time_horizon_counter = 0
        self.global_counter = Gen.current_steps
        self.episode_counter = Gen.current_episodes
        self.log_state = log_state

        if self.log_state is True:
            self.writer = SummaryWriter('./log_tb_files/Single_Battery/'+Gen.TIME_NOW)

        img_size = Gen.Image_size
        cycles_in_act = Gen.cycles_in_act
        self.observation_space = spaces.Box(low=0,high=255,shape=(img_size,img_size,cycles_in_act),dtype=np.uint8)

        self.action_space = spaces.Discrete(Gen.action_number)
        # actions only consider the charging process

        self.action_Cvalue = Gen.action_Cvalue
        self.action_cutoff = Gen.action_cutoff
        self.action_Ivalue = Gen.action_Ivalue

        self.state = None
        self.s_array = Gen.s_array_init
        self.s_array_0 = Gen.s_array_init

        # TensorBoard Logging
        self.tb_input_current = None
        self.tb_input_Crate = None
        self.tb_input_cutoff = None
        # self.tb_state_of_charge = self.SOC_0
        self.tb_reward_list = []
        self.tb_reward_mean = None
        self.tb_reward_sum = None
        self.tb_instantaneous_reward = None


    def step(self, action):
        """Accepts an action and returns a tuple (observation, reward, done, info)"""

        action_num = action.item()
        input_current = self.action_Ivalue[action_num]
        input_Crate = self.action_Cvalue[action_num]
        input_cutoff = self.action_cutoff[action_num]
        # (NOTE: Input Current -> Charging)
        '''option(1) discharging current is fixed
           option(2) [√] discharging current is the same as charging action
        '''
        # dis_current = Gen.discharging_rate * self.I_unit
        dis_current = - input_current

        # Determine if the Episode has Reached it's termination Time
        done_timesteps = bool(self.time_horizon_counter >= self.training_duration)

        done_Stage_decay = all(_ == self.limit_Stage for _ in self.crit_Stage_decay)

        done = done_timesteps or done_Stage_decay

        if not done:
            '''
            action--(device control)--state--(vision model)--reward
            introduce solartron control and state prediction
            '''
            exppath, datapath = pathload(action_num)
            times, currents, voltages = pycontrol_single(exppath,datapath,Virtual=False)  # current -> mA
            data = inst_cycles_process(times, currents, voltages)

            self.s_array = data2shape_array(data)

            # state stage prediction
            StagePred = stagemodel(self.s_array)

            self.crit_Stage_decay.append(StagePred)

            vel_scap_list = velocity_detect(data)

            reward = self.reward_function(StagePred, vel_scap_list, action_num)

            # data saving during cycling process
            step_count = self.global_counter
            save_time = self.save_time
            data_monitor_save(data, action_num, reward, StagePred, vel_scap_list, step_count, save_time)

            # 8 编写reset函数，要运行随机或默认测试实验来得到初始状态

        else:

            logger.warn(
                  "You are calling 'step()' even though this "
                  "environment has already returned done = True.")
            print('=================================')
            print(" || Please replace a new battery for AI module. ||")
            print('=================================')
            reward = 0.0
            sys.exit()

        # Log TensorBoard Variables
        self.tb_input_current = input_current
        self.tb_input_Crate = input_Crate
        self.tb_input_cutoff = input_cutoff
        # self.tb_state_of_charge = self.SOC
        self.tb_reward_list.append(reward)
        self.tb_reward_mean = np.mean(self.tb_reward_list)
        self.tb_reward_sum = np.sum(self.tb_reward_list)

        self.tb_instantaneous_reward = reward
        print('instantaneous_reward:', reward)

        if self.log_state is True:
            self.writer.add_scalar('Input_Current', self.tb_input_current, self.global_counter)
            self.writer.add_scalar('Input_Crate', self.tb_input_Crate, self.global_counter)
            self.writer.add_scalar('Input_Cutoff', self.tb_input_cutoff, self.global_counter)
            # self.writer.add_scalar('Simple/SOC', self.tb_state_of_charge, self.global_counter)
            self.writer.add_scalar('Instantaneous Reward', self.tb_instantaneous_reward, self.global_counter)
            self.writer.add_scalar('Cum. Reward', self.tb_reward_sum, self.global_counter)
            self.writer.add_scalar('Avg. Reward', self.tb_reward_mean, self.global_counter)
            self.writer.add_scalar('Num. Episodes', self.episode_counter, self.global_counter)

        self.time_horizon_counter += 1
        self.global_counter += 1
        self.state = [self.s_array]

        return np.array(self.state), reward, done, {}

    def reset(self):

        print('RL model resetting: initial action')
        action_num = 10 # default action

        print('====== resetting ======')
        exppath, datapath = pathload(action_num)
        times, currents, voltages = pycontrol_single(exppath, datapath, Virtual=False)  # current -> mA

        data = inst_cycles_process(times, currents, voltages)

        self.s_array_0 = data2shape_array(data)
        self.s_array = self.s_array_0

        # reward record
        StagePred = stagemodel(self.s_array)

        vel_scap_list = velocity_detect(data)

        reward = self.reward_function(StagePred, vel_scap_list, action_num)

        data_reset_save(data, action_num, reward, StagePred, vel_scap_list, self.episode_counter, self.save_time)

        self.episode_counter += 1
        self.time_horizon_counter = 0
        # self.crit_SOH = deque([1,1,1], maxlen=3)
        self.crit_Stage_decay = deque([0 for _ in range(10)], maxlen=10)
        self.crit_Stage_decay.append(StagePred)

        self.state = [self.s_array]

        return np.array(self.state)

    def reward_function(self, StagePred, vel_scap_list, action_num):
        """
        (1) Stage reward
        (2) Coincidence reward
        (3) Crate & cutoff reward
        """
        C_rate = self.action_Cvalue[action_num]
        cutoff = self.action_cutoff[action_num]

        max_Crate = max(self.action_Cvalue)
        max_cutoff = max(self.action_cutoff)
        min_Crate = min(self.action_Cvalue)
        min_cutoff = min(self.action_cutoff)

        T_coinc_1 = Gen.T_coinc_1  # completely coincided
        T_coinc_2 = Gen.T_coinc_2  # partly coincided
        T_coinc_3 = Gen.T_coinc_3  # difference

        # 1
        reward_Stage = 0
        reward_Crate = 0
        reward_cutoff = 0
        if StagePred is not None:
            # [0]:stable; [1]:transition; [2]:consistent decay
            if StagePred == 0:
                reward_Stage = 1
                reward_Crate = -0.5 + round((C_rate-min_Crate)/(max_Crate-min_Crate), 1)
                reward_cutoff = -0.5 + round((cutoff-min_cutoff)/(max_cutoff-min_cutoff), 1)
            elif StagePred == 1:
                reward_Stage = -1
            elif StagePred == 2:
                reward_Stage = -5

        # 2
        reward_sCap = 0
        for diff_scap in vel_scap_list[1:]:
            if abs(diff_scap) < T_coinc_1:
                reward_sCap += 0.4
            elif T_coinc_1 <= abs(diff_scap) < T_coinc_2:
                reward_sCap += 0.2
            elif T_coinc_2 <= abs(diff_scap) < T_coinc_3:
                reward_sCap += -0.2
            elif abs(diff_scap) >= T_coinc_3:
                reward_sCap += -0.4
        reward_sCap = round(reward_sCap/len(vel_scap_list[1:]),1)

        reward_total = round(reward_Stage + reward_sCap + reward_Crate + reward_cutoff, 1)

        return reward_total
