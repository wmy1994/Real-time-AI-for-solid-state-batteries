""" Plot of prediction performance
Author@Mingyang
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

'''Plot confusion matrix
'''

file_name = 'resnet50_State_Cls_Test_result.txt'
root = '_model_data'

txt_root = root + '/' + file_name

# colors = np.random.rand(50)

f = open(txt_root, 'r')
data = f.readlines()

true_labels = []
predicted_labels = []
xline = [-200,2500]
yline = [-200,2500]

for line in data:
    word = line.split()

    true_labels.append(int(word[1]))
    predicted_labels.append(int(word[2]))

f.close()


cm = confusion_matrix(true_labels,predicted_labels,)

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

labels = np.unique(true_labels)

plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.RdPu)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

thresh = cm_norm.max() / 2.0
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, "{:.2f}".format(cm_norm[i, j]),
             horizontalalignment='center',
             color='white' if cm_norm[i, j] > thresh else 'black')

plt.show()




