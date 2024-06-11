import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# add1 = np.load('add_index/index_P0_3.npy')
# add2 = np.load('add_index/index_P1_3.npy')
# add3 = np.load('add_index/index_P2_3.npy')
# add4 = np.load('add_index/index_P3_3.npy')
# add5 = np.load('add_index/index_P4_3.npy')
#
# add1 = np.sort(add1)
# add2 = np.sort(add2)
# add3 = np.sort(add3)
# add4 = np.sort(add4)
# add5 = np.sort(add5)
# print(add1)
# print(add2)
# print(add3)
# print(add4)
# print(add5)

M1=[0.8812, 0.9105, 0.8924, 0.8846, 0.8911, 0.8846, 0.9021]
M2=[0.9673, 0.9727,	0.9756, 0.9805,	0.9689,	0.9753, 0.9675]
# M3=[0.9233, 0.9438, 0.9412, 0.9165, 0.9072, 0.9253, 0.8871]

AddNum = [40, 50, 60, 70, 80, 90, 100]

fig = plt.figure(figsize=(6, 4), dpi=400)
plt.grid()
plt.plot(AddNum, M1, label='Experiments on the IP dataset', linewidth='2.0', c='r', marker='.', alpha=0.5)
plt.plot(AddNum, M2, label='Experiments on the SA dataset', linewidth='2.0', c='b', marker='.', alpha=0.5)
# plt.plot(AddNum, M3, label='Experiments on the PU dataset', linewidth='2.0', c='g', marker='.', alpha=0.5)
# 添加点
plt.scatter(x=50, y=0.9105, s=80, c='r', marker='*')
plt.scatter(x=70, y=0.9805, s=80, c='b', marker='*')
# plt.scatter(x=50, y=0.9438, s=80, c='g', marker='*')
# 添加水平线
# plt.plot([45, 50, 55], [0.9105, 0.9105, 0.9105], label='PU', linewidth='2.0', c='r', ls='--', alpha=0.6)
# plt.plot([65, 70, 75], [0.9805, 0.9805, 0.9805], label='PU', linewidth='2.0', c='b', ls='--', alpha=0.6)
# plt.plot([45, 50, 55], [0.9438, 0.9438, 0.9438], label='PU', linewidth='2.0', c='g', ls='--', alpha=0.6)

# # 添加垂直线
# plt.plot([50.2, 50.2], [0.85, 0.9105], linewidth='2.0', c='r', ls='--', alpha=0.6)
# plt.plot([70, 70], [0.85, 0.9805], linewidth='2.0', c='b', ls='--', alpha=0.6)
# plt.plot([49.9, 49.9], [0.85, 0.9438], linewidth='2.0', c='g', ls='--', alpha=0.6)

plt.xticks(AddNum, font='Times New Roman')
plt.yticks(font='Times New Roman')
plt.ylim(0.85, None)
plt.legend(loc="lower right", prop='Times New Roman')
plt.xlabel('Number of GCN training epochs', font='Times New Roman')
plt.ylabel('Accuracy(mean of 5 trials)', font='Times New Roman')
# plt.title('Testing accuracy of HPCAN on three dataset.', font='Times New Roman')

# 显示图像
plt.savefig('fig/discuss3-by.png')
plt.show()
print('~')
