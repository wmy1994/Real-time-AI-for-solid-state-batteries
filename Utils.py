import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms

from Prediction_func.Cls_utils import get_network_Cls

from Parameters import *
plt.switch_backend('agg')

Gen = General_param
Inst_p = Instrument_param
Stage_p = Stage_Param

sys.path.append(r"./Exp_Pyctrl_solartron/Action_exp_setting")
MainPath = Inst_p.MainPath

# different actions
actionname_list = []
for i in range(Gen.action_number):
    actionname_list.append('Action_' + str(i))


class Logger(object):
    def __init__(self,filename='default.log',stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

original_file = "Terminal_log.txt"
file_name, file_ext = os.path.splitext(original_file)
DATE_FORMAT = Gen.DATE_FORMAT
time_point = datetime.now().strftime(DATE_FORMAT)
new_file = f"{file_name}_{time_point}{file_ext}"

sys.stdout = Logger(new_file, sys.stdout)


def pathload(action_num):
    print('====== chosen action:', action_num, '======')
    print('C rate =', Gen.action_Cvalue[action_num],'| cutoff = ', Gen.action_cutoff[action_num])
    print('==================')

    actionpath = os.path.join(MainPath, actionname_list[action_num])
    exppath = os.path.join(actionpath, 'exper.info')
    datapath = os.path.join(actionpath, 'DataFile.data')
    return exppath, datapath


def inst_results_RGBprocess(times=[],currents=[],voltages=[]):
    data = {}
    St = 0
    Ed = 0
    count = 0
    cur_copy = currents.copy()
    for i in range(len(cur_copy)-1):
        if cur_copy[i]*cur_copy[i+1] < 0:
            if Ed == 0:
                Ed = i+1
                cycle_times = times[St:Ed]
                cycle_currents = currents[St:Ed]
                cycle_voltages = voltages[St:Ed]

            else:
                St = Ed
                Ed = i+1
                cycle_times = times[St:Ed]
                cycle_currents = currents[St:Ed]
                cycle_voltages = voltages[St:Ed]

            rest_times = times[Ed:-1]
            rest_currents = currents[Ed:-1]
            rest_voltages = voltages[Ed:-1]

            cycle_times = np.array(cycle_times)
            cycle_currents = np.array(cycle_currents)
            cycle_voltages = np.array(cycle_voltages)

            cycle_times = cycle_times[:, np.newaxis]
            cycle_currents = cycle_currents[:, np.newaxis]
            cycle_voltages = cycle_voltages[:, np.newaxis]

            data[count] = np.concatenate((cycle_times,cycle_currents,cycle_voltages),axis=1)
            count = count + 1

    rest_times = np.array(rest_times)
    rest_currents = np.array(rest_currents)
    rest_voltages = np.array(rest_voltages)
    rest_times = rest_times[:, np.newaxis]
    rest_currents = rest_currents[:, np.newaxis]
    rest_voltages = rest_voltages[:, np.newaxis]
    data[count] = np.concatenate((rest_times,rest_currents,rest_voltages),axis=1)
    # data[0],[1]-> R channel, data[2][3]-> G channel, data[4],[5]-> B channel

    return data


def inst_results_process(times=[],currents=[],voltages=[]):
    data = {}
    St = 0
    Ed = 0
    count = 0
    cur_copy = currents.copy()
    for i in range(len(cur_copy)-1):
        if cur_copy[i]*cur_copy[i+1] < 0:
            Ed = i+1
            cha_times = times[St:Ed]
            cha_currents = currents[St:Ed]
            cha_voltages = voltages[St:Ed]

            dis_times = times[Ed:]
            dis_currents = currents[Ed:]
            dis_voltages = voltages[Ed:]
            break

    cha_times = np.array(cha_times)
    cha_currents = np.array(cha_currents)
    cha_voltages = np.array(cha_voltages)

    cha_times = cha_times[:, np.newaxis]
    cha_currents = cha_currents[:, np.newaxis]
    cha_voltages = cha_voltages[:, np.newaxis]

    data[count] = np.concatenate((cha_times,cha_currents,cha_voltages),axis=1)
    count = count + 1

    dis_times = np.array(dis_times)
    dis_currents = np.array(dis_currents)
    dis_voltages = np.array(dis_voltages)
    dis_times = dis_times[:, np.newaxis]
    dis_currents = dis_currents[:, np.newaxis]
    dis_voltages = dis_voltages[:, np.newaxis]
    data[count] = np.concatenate((dis_times,dis_currents,dis_voltages),axis=1)
    # data[0]-> charging, data[1]-> discharging

    return data



def inst_cycles_process(times=[],currents=[],voltages=[]):
    data = {}
    St = 0
    Ed = 0
    count = 0
    cur_copy = currents.copy()
    for i in range(len(cur_copy)-1):
        if cur_copy[i]*cur_copy[i+1] < 0:
            if Ed == 0:
                Ed = i+1
                cycle_times = times[St:Ed]
                cycle_currents = currents[St:Ed]
                cycle_voltages = voltages[St:Ed]

            else:
                St = Ed
                Ed = i+1
                cycle_times = times[St:Ed]
                cycle_currents = currents[St:Ed]
                cycle_voltages = voltages[St:Ed]

            rest_times = times[Ed:]
            rest_currents = currents[Ed:]
            rest_voltages = voltages[Ed:]

            cycle_times = np.array(cycle_times)
            cycle_currents = np.array(cycle_currents)
            cycle_voltages = np.array(cycle_voltages)

            cycle_times = cycle_times[:, np.newaxis]
            cycle_currents = cycle_currents[:, np.newaxis]
            cycle_voltages = cycle_voltages[:, np.newaxis]

            data[count] = np.concatenate((cycle_times,cycle_currents,cycle_voltages),axis=1)
            count = count + 1

    rest_times = np.array(rest_times)
    rest_currents = np.array(rest_currents)
    rest_voltages = np.array(rest_voltages)
    rest_times = rest_times[:, np.newaxis]
    rest_currents = rest_currents[:, np.newaxis]
    rest_voltages = rest_voltages[:, np.newaxis]
    data[count] = np.concatenate((rest_times,rest_currents,rest_voltages),axis=1)
    # data[even]-> charging ; data[odd]-> discharging

    return data


def data2shape_array(data, Image_size=Gen.Image_size):
    cycles_in_act = int(len(data)/2)

    Qnorm = Gen.Q_plot
    Imax = Gen.Imax
    Imin = Gen.Imin
    Vmax = Gen.Vmax
    Vmin = Gen.Vmin

    sheet_list = []
    for i in range(cycles_in_act):
        sheet_list.append(np.zeros((Image_size,Image_size),dtype=np.uint8))

    data_Q = {}
    # data_Q[even]-> charging capacity ; data_Q[odd]-> discharging capacity

    for i in range(0,len(data),2):
        Qc_value = 0
        Qd_value = 0
        Qc_list = []
        Qd_list = []

        # charging process
        for j in range(len(data[i][:,1])-1):
            dt = data[i][j+1,0] - data[i][j,0]
            Qc_value += abs(dt * ((data[i][j,1])+(data[i][j+1,1]))/2)  # mA*second
            Qc_list.append(Qc_value)

        # discharging process
        for m in range(len(data[i+1][:,1])-1):
            dt = data[i+1][m+1,0] - data[i+1][m,0]
            Qd_value += abs(dt * ((data[i+1][m,1]) + (data[i+1][m+1,1]))/2)  # mA*second
            Qd_list.append(Qd_value)

        data_Q[i] = np.array(Qc_list)
        data_Q[i+1] = np.array(Qd_list)

    for i in range(0, len(data), 2):
        # charging process
        N_Qc_list = data_Q[i]/Qnorm
        N_Ic_list = (data[i][:,1]-Imin)/(Imax-Imin)
        N_Vc_list = (data[i][:,2]-Vmin)/(Vmax-Vmin)

        # discharging process
        N_Qd_list = data_Q[i+1]/Qnorm
        N_Qd_list = data_Q[i+1][-1]/Qnorm - N_Qd_list
        N_Id_list = (data[i+1][:,1]-Imin)/(Imax-Imin)
        N_Vd_list = (data[i+1][:,2]-Vmin)/(Vmax-Vmin)

        sheet_mat_I = np.zeros((int(0.5*Image_size), Image_size),dtype=np.uint8)
        sheet_mat_V = np.zeros((int(0.5*Image_size), Image_size),dtype=np.uint8)
        for z in range(Image_size):
            x1 = z / Image_size
            x2 = (z+1) / Image_size

            # charging process
            for k in range(len(N_Qc_list)):
                if x1 <= N_Qc_list[k] < x2:
                    Image_y_Ic = (0.25*Image_size - 1) + round((0.25*Image_size - 1) * N_Ic_list[k])

                    Image_y_Vc = round((0.5*Image_size - 1) * N_Vc_list[k])
                    if Image_y_Ic > 0.5*Image_size - 1 or Image_y_Ic < 0:
                        continue
                    elif Image_y_Vc > 0.5*Image_size - 1 or Image_y_Vc < 0:
                        continue
                    Image_y_Ic = int(0.5*Image_size-1 - Image_y_Ic)
                    Image_y_Vc = int(0.5*Image_size-1 - Image_y_Vc)

                    sheet_mat_I[Image_y_Ic, z] = 255
                    sheet_mat_V[Image_y_Vc, z] = 255

            # discharging process
            for k in range(len(N_Qd_list)):
                if x1 <= N_Qd_list[k] < x2:
                    Image_y_Id = (0.25*Image_size - 1) + round((0.25*Image_size - 1) * N_Id_list[k])

                    Image_y_Vd = round((0.5 * Image_size - 1) * N_Vd_list[k])
                    if Image_y_Id > 0.5*Image_size - 1 or Image_y_Id < 0:
                        continue
                    elif Image_y_Vd > 0.5*Image_size - 1 or Image_y_Vd < 0:
                        continue
                    Image_y_Id = int(0.5*Image_size-1 - Image_y_Id)
                    Image_y_Vd = int(0.5*Image_size-1 - Image_y_Vd)

                    sheet_mat_I[Image_y_Id, z] = 255
                    sheet_mat_V[Image_y_Vd, z] = 255

        sheet_mat_all = np.vstack((sheet_mat_I,sheet_mat_V))

        sheet_list[(i//2)] = sheet_mat_all

    shape_array = np.dstack(sheet_list)

    return shape_array


# Stage prediction part
def stagemodel(RGB, args=Stage_p):
    """
    :param RGB: shape-feature image for battery cycling (np.array)
    :param args: prediction model settings
    :return: predicted state stage of battery: [0]:stable; [1]:transition; [2]:fast decay
    """
    net = get_network_Cls(args)
    net_name = args.net

    tss = transforms.Compose([
        transforms.ToTensor(), ])

    img = RGB  # RGB should be np.array
    img = tss(img)

    img = torch.unsqueeze(img, 0)

    # Stage prediction
    net.load_state_dict(torch.load(args.weights_Stage))
    net.eval()
    with torch.no_grad():
        if args.gpu:
            img = img.cuda()
        output_Stage = net(img)
        _, StageLabel = output_Stage.max(1)
        StagePred = StageLabel.item()

    return StagePred


# evaluate coincidence of the discharging voltage curve
def velocity_detect(data):
    weight = Gen.weight
    cycles_in_act = int(len(data) / 2)
    comp_Scap = []
    for i in range(0, len(data), 2):
        Qsd_value = 0
        Qsd_list = [0]

        # discharging process
        for m in range(len(data[i + 1][:, 1]) - 1):
            dt = data[i + 1][m + 1, 0] - data[i + 1][m, 0]
            # Specific Discharging Capacity (mAh/g)
            Qsd_value += abs(dt * ((data[i + 1][m, 1]) + (data[i + 1][m + 1, 1])) / 2) / 3600 / (weight * 1e-3)
            Qsd_list.append(Qsd_value)
        final_Qsd_value = round(Qsd_value, 2)

        cycle_num = i // 2 + 1

        cha_vol_list = data[i][:, 2]
        dis_vol_list = data[i + 1][:, 2]

        # coincide evaluation of discharging curve
        x_A = Qsd_list[-1]
        x_B = Qsd_list[-2]
        y_A = dis_vol_list[-1]
        y_B = dis_vol_list[-2]
        ref_V = Gen.Vmin

        ref_Scap = (x_A - x_B) / (y_A - y_B) * (ref_V - y_B) + x_B
        comp_Scap.append(ref_Scap)

    vel_scap_list = [0] # step内第1个cycle对应，单值计算，所以赋值0，保留位置
    for j in range(cycles_in_act-1):
        velocity = round(comp_Scap[j+1] - comp_Scap[j], 2)
        vel_scap_list.append(velocity)

    return vel_scap_list


def data_monitor_save(data, action_num, reward, StagePred, vel_scap_list, step_count, save_time):
    weight = Gen.weight
    rootpath = 'post_process'
    folderpath = os.path.join(rootpath, 'monitor_save-' + save_time)
    subpath_np = os.path.join(folderpath,'npdata')
    subpath_img = os.path.join(folderpath,'cycling plot')
    subname_txt = 'cycling data.txt'
    subname_np = 'cycling_data-step_' + str(step_count)
    subname_img = 'performance-step_' + str(step_count) + '.jpg'
    if not os.path.exists(subpath_np):
        os.makedirs(subpath_np)
    if not os.path.exists(subpath_img):
        os.makedirs(subpath_img)

    save_np = os.path.join(subpath_np, subname_np)
    np.savez(save_np, cycles=data)

    txtpath = os.path.join(folderpath, subname_txt)

    file = open(txtpath, "a")

    is_empty = os.stat(txtpath).st_size == 0
    if is_empty:
        content = 'step\t'+'action num\t'+'ins_reward\t'+'Stage\t'+'Crate\t'+'cutoff\t' + \
                  'cycle num\t'+'Scap cha\t'+'Scap dis\t'+'vel_scap\n'
        file.write(content)

    fig1, ax1 = plt.subplots()

    for i in range(0, len(data), 2):
        Qsc_value = 0
        Qsd_value = 0

        Qsc_list = [0]
        Qsd_list = [0]

        # charging process
        for j in range(len(data[i][:, 1]) - 1):
            dt = data[i][j + 1, 0] - data[i][j, 0]
            # Specific Charging Capacity (mAh/g)
            Qsc_value += abs(dt * ((data[i][j, 1]) + (data[i][j + 1, 1])) / 2) / 3600 / (weight*1e-3)
            Qsc_list.append(Qsc_value)
        final_Qsc_value = round(Qsc_value, 2)

        # discharging process
        for m in range(len(data[i + 1][:, 1]) - 1):
            dt = data[i + 1][m + 1, 0] - data[i + 1][m, 0]
            # Specific Discharging Capacity (mAh/g)
            Qsd_value += abs(dt * ((data[i + 1][m, 1]) + (data[i + 1][m + 1, 1])) / 2) / 3600 / (weight*1e-3)
            Qsd_list.append(Qsd_value)
        final_Qsd_value = round(Qsd_value, 2)

        cycles_in_act = int(len(data)/2)
        cycle_num = step_count*cycles_in_act + i//2 + 1
        Crate = Gen.action_Cvalue[action_num]
        cutoff = Gen.action_cutoff[action_num]

        string = str(step_count)+'\t'+str(action_num)+'\t'+str(reward)+'\t'+str(StagePred)+'\t'+str(Crate)+'\t' + \
                 str(cutoff)+'\t'+str(cycle_num)+'\t'+str(final_Qsc_value)+'\t'+str(final_Qsd_value)+'\t'+str(vel_scap_list[i//2])

        file.write(string + '\n')
        cha_vol_list = data[i][:, 2]
        dis_vol_list = data[i+1][:, 2]
        ax1.plot(Qsc_list, cha_vol_list, label='cyc'+str(cycle_num)+'-cha-'+str(Crate)+'C-'+str(cutoff)+'V', alpha=0.8)
        ax1.plot(Qsd_list, dis_vol_list, label='cyc'+str(cycle_num)+'-dis-'+str(Crate)+'C-'+str(cutoff)+'V', alpha=0.8)

    plt.legend()
    plt.grid(True)
    ax1.set_xlabel('Specific Capacity (mAh/g)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('cycling performance for step' + str(step_count))

    save_img = os.path.join(subpath_img,subname_img)
    fig1.savefig(save_img, format='jpg')

    plt.close('all')
    file.close()


def data_reset_save(data, action_num, reward, StagePred, vel_scap_list, episode_counter, save_time):
    weight = Gen.weight
    rootpath = 'post_process'
    folderpath = os.path.join(rootpath,'monitor_reset_save-' + save_time)
    subpath_np = os.path.join(folderpath,'npdata')
    subpath_img = os.path.join(folderpath,'cycling plot')
    subname_txt = 'reset-cycling data.txt'
    subname_np = 'reset-cycling_data-episode_' + str(episode_counter)
    subname_img = 'reset-performance-episode_' + str(episode_counter) + '.jpg'
    if not os.path.exists(subpath_np):
        os.makedirs(subpath_np)
    if not os.path.exists(subpath_img):
        os.makedirs(subpath_img)

    save_np = os.path.join(subpath_np, subname_np)
    np.savez(save_np, cycles=data)

    txtpath = os.path.join(folderpath, subname_txt)

    file = open(txtpath, "a")
    is_empty = os.stat(txtpath).st_size == 0
    if is_empty:
        content = 'step\t' + 'action num\t' + 'ins_reward\t' + 'Stage\t' + 'Crate\t' + 'cutoff\t' + \
                  'cycle num\t' + 'Scap cha\t' + 'Scap dis\t' + 'vel_scap\n'
        file.write(content)

    fig1, ax1 = plt.subplots()

    for i in range(0, len(data), 2):
        Qsc_value = 0
        Qsd_value = 0
        Qsc_list = [0]
        Qsd_list = [0]

        # charging process
        for j in range(len(data[i][:, 1]) - 1):
            dt = data[i][j + 1, 0] - data[i][j, 0]
            # Specific Charging Capacity (mAh/g)
            Qsc_value += abs(dt * ((data[i][j, 1]) + (data[i][j + 1, 1])) / 2) / 3600 / (weight*1e-3)
            Qsc_list.append(Qsc_value)
        final_Qsc_value = round(Qsc_value, 2)

        # discharging process
        for m in range(len(data[i + 1][:, 1]) - 1):
            dt = data[i + 1][m + 1, 0] - data[i + 1][m, 0]
            # Specific Discharging Capacity (mAh/g)
            Qsd_value += abs(dt * ((data[i + 1][m, 1]) + (data[i + 1][m + 1, 1])) / 2) / 3600 / (weight*1e-3)
            Qsd_list.append(Qsd_value)
        final_Qsd_value = round(Qsd_value, 2)

        cycles_in_act = int(len(data)/2)
        cycle_num = i//2 + 1
        Crate = Gen.action_Cvalue[action_num]
        cutoff = Gen.action_cutoff[action_num]

        string = str(episode_counter)+'\t'+str(action_num)+'\t'+str(reward)+'\t'+str(StagePred)+'\t'+str(Crate)+'\t'+\
                 str(cutoff)+'\t'+str(cycle_num)+'\t'+str(final_Qsc_value)+'\t'+str(final_Qsd_value)+'\t'+str(vel_scap_list[i//2])

        file.write(string + '\n')
        cha_vol_list = data[i][:, 2]
        dis_vol_list = data[i+1][:, 2]
        ax1.plot(Qsc_list, cha_vol_list, label='cyc'+str(cycle_num)+'-cha-'+str(Crate)+'C-'+str(cutoff)+'V', alpha=0.8)
        ax1.plot(Qsd_list, dis_vol_list, label='cyc'+str(cycle_num)+'-dis-'+str(Crate)+'C-'+str(cutoff)+'V', alpha=0.8)

    plt.legend()
    plt.grid(True)
    ax1.set_xlabel('Specific Capacity (mAh/g)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('reset cycling performance for episode' + str(episode_counter))

    save_img = os.path.join(subpath_img,subname_img)
    fig1.savefig(save_img, format='jpg')

    plt.close('all')
    file.close()
