import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns

trainFeat = np.load('new_add/labeledF2.npy')
Pindex = np.load('new_add/prototype2.npy')
label = np.load('new_add/Labeled_L2.npy')

Index = np.argsort(label)
newPindex =[]
for i in range(len(Pindex)):
    newIndex = np.where(Index == Pindex[i])[0]
    newPindex.append(newIndex)
newPindex = np.array(newPindex).reshape(-1)
trainFeat = trainFeat[Index]
label = label[Index]
pca = PCA(n_components=2)
pca.fit(trainFeat)
reduced_data = pca.transform(trainFeat)
color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'orange', 'brown', 'skyblue', 'crimson', 'teal', 'violet', 'indigo', 'slateblue', 'mediumseagreen', 'darkgoldenrod']
fig = plt.figure(figsize=(4, 4), dpi=400)
add_number = -5
# 绘制散点图并添加编号
for j in range(len(reduced_data)):
    color_index = int(label[j])
    if j in newPindex:
        classIndex = np.where(newPindex == j)[0]
        plt.scatter(reduced_data[j][0], reduced_data[j][1], marker='*', s=50, color=color_map[color_index], alpha=0.9, label='Prototype'+str(int(classIndex+1)))
        if classIndex == 1:
            add_numberx = 20
            add_numbery = 1
            plt.annotate(int(Pindex[classIndex]), (reduced_data[j][0], reduced_data[j][1]), font='Times New Roman',
                         fontsize=12, ha='center', va='bottom',
                         xytext=(reduced_data[j][0] + add_numberx, reduced_data[j][1] + add_numbery),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', facecolor=color_map[color_index]))
        elif classIndex == 2:
            add_numberx = -20
            add_numbery = -3
            plt.annotate(int(Pindex[classIndex]), (reduced_data[j][0], reduced_data[j][1]), font='Times New Roman',
                         fontsize=12, ha='center', va='bottom',
                         xytext=(reduced_data[j][0] + add_numberx, reduced_data[j][1] + add_numbery),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', facecolor=color_map[color_index]))
        elif classIndex == 3:
            add_numberx = -20
            add_numbery = -10
            plt.annotate(int(Pindex[classIndex]), (reduced_data[j][0], reduced_data[j][1]), font='Times New Roman',
                         fontsize=12, ha='center', va='bottom',
                         xytext=(reduced_data[j][0] + add_numberx, reduced_data[j][1] + add_numbery),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',facecolor=color_map[color_index]))
        elif classIndex == 4:
            add_numberx = 15
            add_numbery = 3
            plt.annotate(int(Pindex[classIndex]), (reduced_data[j][0], reduced_data[j][1]), font='Times New Roman',
                         fontsize=12, ha='center', va='bottom',
                         xytext=(reduced_data[j][0] + add_numberx, reduced_data[j][1] + add_numbery),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', facecolor=color_map[color_index]))
        elif classIndex == 5:
            add_numberx = -20
            add_numbery = -3
            plt.annotate(int(Pindex[classIndex]), (reduced_data[j][0], reduced_data[j][1]), font='Times New Roman',
                         fontsize=12, ha='center', va='bottom',
                         xytext=(reduced_data[j][0] + add_numberx, reduced_data[j][1] + add_numbery),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                         facecolor=color_map[color_index]))
        elif classIndex == 6:
            add_numberx = 15
            add_numbery = -5
            plt.annotate(int(Pindex[classIndex]), (reduced_data[j][0], reduced_data[j][1]), font='Times New Roman',
                         fontsize=12, ha='center', va='bottom',
                         xytext=(reduced_data[j][0] + add_numberx, reduced_data[j][1] + add_numbery),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', facecolor=color_map[color_index]))
        elif classIndex == 9:
            add_numberx = 1
            add_numbery = -10
            plt.annotate(int(Pindex[classIndex]), (reduced_data[j][0], reduced_data[j][1]), font='Times New Roman',
                         fontsize=12, ha='center', va='bottom',
                         xytext=(reduced_data[j][0] + add_numberx, reduced_data[j][1] + add_numbery),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', facecolor=color_map[color_index]))
        else:
             plt.annotate(int(Pindex[classIndex]), (reduced_data[j][0], reduced_data[j][1]), font='Times New Roman',
                         fontsize=12)
    else:
        plt.scatter(reduced_data[j][0], reduced_data[j][1], marker='o', s=20, color=color_map[color_index], alpha=0.2)

plt.legend(loc="center right", prop='Times New Roman', bbox_to_anchor=(1.5, 0.5))
plt.title('After Stage II', font='Times New Roman', fontsize=18)
# 去掉刻度
plt.xticks([])
plt.yticks([])
plt.savefig('new_add/stage2-2.png', bbox_inches='tight')
# 显示图形
plt.show()