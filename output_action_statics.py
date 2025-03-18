"""
Distribution for all Actions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
from Parameters import General_param
from mpl_toolkits.mplot3d import Axes3D

# Extract information from monitor_save.txt
folder_timepoint = '08_September_2024_20h_27m'

folder_path = 'post_process\\monitor_save-' + folder_timepoint
txtPath = os.path.join(folder_path, 'cycling data.txt')

setting_Cvalue = [15,16,17,18,19,20]
setting_cutoff = [4.2,4.25,4.3]

cycles_list = []
Crate_list = []
cutoff_list = []
Scap_cha_list = []
Scap_dis_list = []
Stage_list = []
action_num_list = [0 for i in range(len(setting_Cvalue)*len(setting_cutoff))]
action_crate_list = [i for i in setting_Cvalue for j in setting_cutoff]
action_cut_list = [j for i in setting_Cvalue for j in setting_cutoff]
step = -1
with open(txtPath,'r') as f:
    lines = f.readlines()
    for line in lines:
        if line == lines[0]:
            continue
        line_values = line.split('\t')
        # 一定要记得，txt读取后，字符转数值
        step_new = int(line_values[0])
        if step == step_new:
            continue
        step = step_new
        Crate_list.append(int(line_values[4]))
        cutoff_list.append(float(line_values[5]))
        cycles_list.append(int(line_values[6]))
        # Scap_cha_list.append(float(line_values[7]))
        Scap_dis_list.append(float(line_values[8]))
        Stage_list.append(int(line_values[3]))

action_crate_arr = np.array(action_crate_list)
action_cut_arr = np.array(action_cut_list)
for x,y in zip(Crate_list,cutoff_list):
    index = np.where((action_crate_arr == x) & (action_cut_arr == y))
    # print(index[0])
    action_num_list[int(index[0])] += 1

fig1, ax1 = plt.subplots(1, subplot_kw={'projection': '3d'})
bottom = [0 for _ in action_num_list]  # 设置柱状图的底端位值
colors = sns.color_palette("Set3", n_colors=len(action_num_list))

ax1.bar3d(action_crate_list, action_cut_list, bottom, dx=0.5, dy=0.01, dz=action_num_list, shade=False,edgecolor='gray',color=colors)

ax1.view_init(elev=25, azim=22)

ax1.set_xlabel('C rate')
ax1.set_ylabel('Cutoff')
# ax1.set_yticks((4.2,4.25,4.3))

ax1.set_zlabel('Executed action number')
ax1.set_title('Total Steps = {}'.format(int(len(Crate_list))))

plt.show()