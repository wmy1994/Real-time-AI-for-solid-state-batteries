"""
PCA space
"""
import os.path

import matplotlib.pyplot as plt
import numpy as np
import sys
from PIL import Image
from Utils import *
from sklearn.decomposition import PCA
from matplotlib.ticker import FuncFormatter

folder_timepoint = '30_August_2024_09h_15m'


filePath = 'post_process\\monitor_save-' + folder_timepoint + '\\npdata'

steps_used = -1

savePath = 'post_process\\monitor_save-' + folder_timepoint + '\\PCA_result'
ifsaveimg = True

allsteps_features = []
allcycles_features = []
everystep_features = []
npz_files = os.listdir(filePath)
npz_files.sort(key=lambda x: int(x.split('_')[2][:-4]))
total_step = len(npz_files)
if steps_used == -1:
    steps_used = total_step
step_num = 0
for index, npzname in enumerate(npz_files):
    if index+1 > steps_used:
        break
    npzPath = os.path.join(filePath, npzname)
    data_npz = np.load(npzPath, allow_pickle=True)
    # 逐个输出一个.npz文件中的所有数组
    for file_name in data_npz.files:
        data_dict = data_npz[file_name].tolist()
        # print(f"{file_name}:")
        # print(data)
        # print('==========')
        data_list = []
        for i in data_dict:
            # print(data_array[index])
            data_list.append(data_dict[i])

        # npz需要array->dict->list
        data = data_list

        s_array = data2shape_array(data)
        image_array = s_array.flatten()
        allsteps_features.append(image_array)

        _0_array = s_array[:, :, 0].flatten()
        _1_array = s_array[:, :, 1].flatten()
        _2_array = s_array[:, :, 2].flatten()
        allcycles_features.append(_0_array)
        allcycles_features.append(_1_array)
        allcycles_features.append(_2_array)


fig1, axs1 = plt.subplots(1,2)
axs1_list = axs1.flatten()
ax0 = axs1_list[0]
ax1 = axs1_list[1]

pca = PCA(n_components=5)
pca_features = pca.fit_transform(allsteps_features)
PC1 = np.array(pca_features[:, 0])
PC2 = np.array(pca_features[:, 1])
PC3 = np.array(pca_features[:, 2])

PCA_space0 = ax0.scatter(PC1, PC2, c=range(len(PC1)), cmap='viridis')
ax0.set_xlabel('PC1')
ax0.set_ylabel('PC2')
# ax0.set_title('PCA - all steps')
ax0.set_title('PC2 vs. PC1 at step {}/{}'.format(str(steps_used), str(total_step)))
colorbar_0 = plt.colorbar(PCA_space0)
colorbar_0.set_label('Step index')
PCA_space1 = ax1.scatter(PC1, PC3, c=range(len(PC1)), cmap='viridis')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC3')
# ax1.set_title('PCA - all steps')
ax1.set_title('PC3 vs. PC1 at step {}/{}'.format(str(steps_used), str(total_step)))
colorbar_1 = plt.colorbar(PCA_space1)
colorbar_1.set_label('Step index')
fig1.set_size_inches(14, 6)
fig1.subplots_adjust(left=0.1, right=1, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)

if ifsaveimg:
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    figname = os.path.join(savePath, 'PCA_steps-S_'+str(steps_used)+'.jpg')
    fig1.savefig(figname, format='jpg', dpi=200)
    plt.close('all')
