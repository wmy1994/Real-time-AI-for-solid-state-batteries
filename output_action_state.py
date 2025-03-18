"""
Changes in capacity standards corresponding to each action (Crate, Cutoff) separately (unified benchmark)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
from Parameters import General_param

# Extract information from monitor_save.txt
folder_timepoint = '08_September_2024_20h_27m'


folder_path = 'post_process\\monitor_save-' + folder_timepoint
txtPath = os.path.join(folder_path, 'cycling data.txt')

setting_Cvalue = [15,16,17,18,19,20]
setting_cutoff = [4.2,4.25,4.3]

BM_Crate = 18
BM_cutoff = 4.25


cycles_list = []
Crate_list = []
cutoff_list = []
Scap_cha_list = []
Scap_dis_list = []
Stage_list = []
with open(txtPath,'r') as f:
    lines = f.readlines()
    for line in lines:
        if line == lines[0]:
            continue
        line_values = line.split('\t')
        Crate_list.append(int(line_values[4]))
        cutoff_list.append(float(line_values[5]))
        cycles_list.append(int(line_values[6]))
        # Scap_cha_list.append(float(line_values[7]))
        Scap_dis_list.append(float(line_values[8]))
        Stage_list.append(int(line_values[3]))

fig1, ax1 = plt.subplots()

# cycles_list, Crate_list, cutoff_list, Scap_dis_list
action_state_dict = {}
action_cycle_dict = {}
cycle_final, retention_sum = 0, 0

for x in setting_Cvalue:
    for y in setting_cutoff:
        cyc_list = []
        state_list = []
        for index, cycle in enumerate(cycles_list):
            if Crate_list[index] == x and cutoff_list[index] == y:
                cyc_list.append(cycle)
                state_list.append(Scap_dis_list[index])
        keyname = str(x)+'C-'+str(y)+'V'
        action_cycle_dict[keyname] = cyc_list
        action_state_dict[keyname] = state_list

avg_retention = round(retention_sum/len(setting_Cvalue)/len(setting_cutoff), 4)
print('\nfinal cycle = {} | retention ratio: {}%'.format(cycle_final, avg_retention*100))

for key, state in action_state_dict.items():
    ax1.plot(action_cycle_dict[key],state,alpha=0.5,label=key)
    if key == str(BM_Crate)+'C-'+str(BM_cutoff)+'V':
        BM_line = ax1.scatter(action_cycle_dict[key], state, alpha=0.8, label=key)
        plt.legend(handles=[BM_line])

ax1.set_ylim(0, 200)
ax1.set_xlabel('Cycle index')
ax1.set_ylabel('Specific Capacity (mAh/g)')
ax1.set_title('Capacity retention performance')
# plt.legend()
plt.grid()

plt.show()


