import numpy as np
from datetime import datetime

"""This is the file which stores Battery Parameters for Reinforcement learning
"""

# General parameters of RL for solid-state battery
class General_param:
    # RL setting
    # action_number: 'Action_0' ~ 'Action_(number-1)'

    Image_size = 112
    cycles_in_act = 3
    Cmax = 30  # max Crate in shape image
    weight = 0.816  # loading weight (mg)
    Q_unit = 150  # Specific Capacity (mAh/g)
    Q_plot_ratio = 1.2  # plot ratio of shape image

    total_timesteps = 15000
    training_duration = 5000
    # counter setting for training (default)
    current_steps = 0
    current_episodes = 0

    '''battery info: relative Capacity for different action
    '''
    setting_Cvalue = [15,16,17,18,19,20]
    setting_cutoff = [4.2, 4.25, 4.3]

    # Action:
    action_Cvalue = []
    action_cutoff = []
    for Cvalue in setting_Cvalue:
        for cutoff in setting_cutoff:
            action_Cvalue.append(Cvalue)
            action_cutoff.append(cutoff)
    print('action Crate', action_Cvalue)
    print('action cutoff', action_cutoff)
    action_number = len(action_Cvalue)

    # capacity cutoff setting
    action_time = [int(3600/C) for C in action_Cvalue]  # unit: second
    print('action time', action_time)

    ''' shape array parameters'''
    capacity_mAh = Q_unit * weight * 1e-3  # Q_unit * weight
    I_unit = capacity_mAh / 1  # current(mA) of 1C, (mAh / 1h -> mA*s / 3600s --> mA)
    capacity = capacity_mAh * 3600  # (mA*s)
    Q_plot = Q_plot_ratio * capacity

    action_Ivalue = []
    for i in range(len(action_Cvalue)):
        action_Ivalue.append(round(action_Cvalue[i] * I_unit,2))
    print('action_Ivalue:', action_Ivalue)

    # current(mA)-Crate
    Imax = Cmax * I_unit  # maximum current value
    Imin = 0
    # voltage window
    Vmax = 5
    Vmin = 2

    '''for stopping battery cycling'''
    limit_Stage = 2  # decay state == 2

    s_array_init = np.zeros((Image_size,Image_size,cycles_in_act),dtype=np.uint8)

    # coincide threshold by velocity of specific capacity
    T_coinc_1 = 0.2
    T_coinc_2 = 0.5
    T_coinc_3 = 1

    # time of we run the script
    DATE_FORMAT = '%d_%B_%Y_%Hh_%Mm'
    TIME_NOW = None

    ########################
    dt = 1  # experimental sampling per 1 second
    T = 328.15  # Ambient Temp. (kelvin) 55 celsius

class Instrument_param:
    # Instrument parameters of experiment control for solid-state battery

    IPaddress = ["169.254.254.11", "169.254.254.10"]
    IPaddress_2 = ["169.254.254.3", "169.254.254.4"]
    # experiment settings
    MainPath = './Exp_Pyctrl_solartron/Action_exp_setting/Action'

class Stage_Param:
    net = 'resnet50'
    weights_Stage = './Prediction_func/weights/Stage/Cap_V_A_label/resnet50-28Bat-best.pth'
    # net = 'shufflenetv2'
    gpu = True










