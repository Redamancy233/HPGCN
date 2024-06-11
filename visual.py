import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

M1=[67.74, 76.94, 80.34, 84.00, 85.94, 86.60]
M2=[67.74, 68.55, 78.02, 83.75, 85.25, 85.54]
M3=[67.74, 66.14, 69.10, 71.72, 73.06, 73.68]
M4=[67.74, 77.65, 80.44, 81.71, 86.21, 86.97]
M5=[67.74, 77.03, 79.02, 79.05, 83.08, 85.33]
M6=[67.74, 78.57, 81.39, 84.38, 85.80, 86.59]
M7=[67.74, 76.49, 79.50, 82.14, 85.79, 87.19]

AddNum = [80, 98, 116, 134, 152, 170]



fig = plt.figure(figsize=(6, 4), dpi=600)
plt.grid()
plt.plot(AddNum, M1, label='Random Sampling', c='palevioletred', marker='o', alpha=0.6)
plt.plot(AddNum, M2, label='Entropy', c='b', marker='o', alpha=0.6)
plt.plot(AddNum, M3, label='CoreSet', c='g', marker='o', alpha=0.6)
plt.plot(AddNum, M4, label='BADGE', c='darkkhaki', marker='o', alpha=0.6)
plt.plot(AddNum, M5, label='UncertainGCN', c='teal', marker='o', alpha=0.6)
plt.plot(AddNum, M6, label='CoreGCN', c='mediumpurple', marker='o', alpha=0.6)
plt.plot(AddNum, M7, label='The proposed method', c='r', marker='o', alpha=0.6)
plt.xticks(AddNum, font='Times New Roman')
plt.yticks(font='Times New Roman')
plt.legend(loc="lower right", prop='Times New Roman')
plt.xlabel('Number of labelled samples', font='Times New Roman')
plt.ylabel('Accuracy(mean of 5 trials)', font='Times New Roman')
plt.title('Testing accuracy of 3D-CNN on PU dataset.', font='Times New Roman')

# 显示图像
plt.savefig('fig/PU/3DCNN-1213.png')
plt.show()
print('~')