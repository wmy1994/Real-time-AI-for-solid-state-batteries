"""
Reading the specific capacity from the txt file, and plot the capacity curve for cycling test
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.ticker import MaxNLocator
import seaborn as sns
color_palette = sns.color_palette(n_colors=3)

# Extract information from monitor_save.txt
folder_timepoint = '08_September_2024_20h_27m'


limit_SOH = -1  # -1 means no limit (0~1)
end_cycle = -1

folder_path = 'post_process\\monitor_save-' + folder_timepoint

txtPath = os.path.join(folder_path, 'cycling data.txt')

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

fig, axs = plt.subplots(1, 2)
axs_list = axs.flatten()
ax0 = axs_list[0]
ax1 = axs_list[1]

# plot 1
ax0.set_title('Capacity retention performance')
ax0.set_ylim(0, 200)
plot_0 = ax0.scatter(cycles_list, Scap_dis_list, c=Stage_list, cmap='viridis', label='Discharging',s=15,alpha=0.9)
ax0.set_xlabel('Cycle index')
ax0.set_ylabel('Specific capacity (mAh/g)')
cbar = plt.colorbar(plot_0)
cbar.set_label('state stage ')
cbar.locator = MaxNLocator(integer=True)
cbar.update_ticks()

# Data Trend Fitting
coefficients = np.polyfit(cycles_list, Scap_dis_list, 3)
fit = np.poly1d(coefficients)
y_fit = fit(cycles_list)
ax0.plot(cycles_list, y_fit, color='red', label='Fit Curve-poly')

step_index = len(Crate_list)//3
total_cycles = len(cycles_list)
print('total cycles =',total_cycles)
if end_cycle == -1:
    end_cycle = total_cycles

'''statistic'''
if 0 < limit_SOH < 1:
    print('limit_SOH = ' + str(limit_SOH))
    retention_list = [round(cap/max(y_fit), 4) for cap in y_fit]
    cyc_index = next((i for i, x in enumerate(retention_list) if x < limit_SOH
                      and (i == len(retention_list) - 1 or retention_list[i+1] < limit_SOH)), None)
    if cyc_index is not None:
        end_cycle = cyc_index
    else:
        print('Still keeping acceptable SOH')

retention = round(y_fit[end_cycle-1]/max(y_fit), 4)
print('cycle = {} | retention ratio: {}%'.format(end_cycle, retention*100))

accum_energy = round(sum(Scap_dis_list[:end_cycle]), 2)
accum_energy_aver = round(sum(Scap_dis_list[:end_cycle])/len(Scap_dis_list[:end_cycle]), 2)
print('accumulated energy = {} | average = {}'.format(accum_energy, accum_energy_aver))

# 充电用时
time_cha_list = [60/Cvalue for Cvalue in Crate_list]
time_cha_total = sum(time_cha_list[:end_cycle])
time_cha_aver = round(sum(time_cha_list[:end_cycle]) / len(Crate_list[:end_cycle]), 2)
print('total charging time (by Crate) = {} | average = {}'.format(time_cha_total, time_cha_aver))

# plot 2 
ax1.set_title('Action strategies')
ax1.axhline(y=4, color='black', linestyle=':', label=' ')
ax1.axhline(y=13, color='black', linestyle=':', label=' ')
ax1.set_xlabel('Cycle index')

ax1.set_ylabel('C rata', color='red')
ax1.yaxis.set_label_coords(-0.07, 0.84)
ax1.set_ylim(-5, 22)
ax1.set_yticks((15, 20))
ax1.plot(cycles_list, Crate_list, linestyle='--', color='red', label='Crate')

ax2 = ax1.twinx()
ax2.set_ylabel('Cut-off voltage (V)', color='blue')
ax2.set_ylim(4, 4.5)
ax2.set_yticks((4.2,4.25,4.3))
ax2.plot(cycles_list, cutoff_list, linestyle='-.', color='blue', label='Cutoff')

ax3 = ax1.twinx()
ax3.set_ylabel('State stage', color='green')
ax3.yaxis.set_label_coords(1.25, 0.17)
ax3.set_ylim(-1, 11)
ax3.set_yticks((0,1,2))
ax3.spines['right'].set_position(('outward', 60)) 
ax3.plot(cycles_list, Stage_list, linestyle='-', color='green', label='Stage')


fig.set_size_inches(12, 5)
fig.subplots_adjust(left=0.05, right=0.85, bottom=0.1, top=0.9, wspace=0.2, hspace=0)
plt.show()

# save_img = os.path.join(subpath_img,subname_img)
# fig.savefig(save_img, format='jpg')

