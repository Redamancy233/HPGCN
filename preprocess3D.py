import gc

import scipy.io as scio
import random
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from function import *
from config import *


# 对高光谱数据进行降维-PCA
def apply_PCA(data, numComponents):

    # PCA需要二维形式，对数据变形
    newData = np.reshape(data, (-1, data.shape[2]))

    # PCA降维
    pca = PCA(n_components=numComponents, whiten=True)
    dataPCA = pca.fit_transform(newData)

    # 将数据还原成三维
    newDataPCA = np.reshape(dataPCA, (data.shape[0], data.shape[1], numComponents))

    return newDataPCA

# Padding
def padding_Zeros(data, margin):
    newData = np.zeros((data.shape[0] + 2 * margin, data.shape[1] + 2 * margin, data.shape[2]))
    xOffset = margin
    yOffset = margin
    newData[xOffset:data.shape[0] + xOffset, yOffset:data.shape[1] + yOffset, :] = data
    return newData


# 在每个像素周围提取 patch
def get_Cubes(data, label, margin, windowSize, removeZeroLabels = True):

    patchesLabels = np.zeros((data.shape[0] * data.shape[1]))
    r0_list = []

    patchesIndex = 0
    for r in range(margin, data.shape[0] - margin):
        for c in range(margin, data.shape[1] - margin):
            patchesLabels[patchesIndex] = label[r-margin, c-margin]
            if patchesLabels[patchesIndex] > 0:
                r0_list.append(patchesIndex)
            patchesIndex = patchesIndex + 1

    newData_len = len(r0_list)
    # split patches
    patchesData = np.zeros((newData_len, windowSize, windowSize, data.shape[2]))


    # 获取cubes
    pIndex = 0
    trueIndex = 0
    for rr in range(margin, data.shape[0] - margin):
        for cc in range(margin, data.shape[1] - margin):
            if pIndex in r0_list:
                patch = data[rr - margin:rr + margin + 1, cc - margin:cc + margin + 1]
                patchesData[trueIndex, :, :, :] = patch
                trueIndex = trueIndex + 1
            pIndex = pIndex + 1


    # 去0标签
    if removeZeroLabels:
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1

    return patchesData, patchesLabels

def get_sample(data, label, sampleNum):
    index = [x for x in range(len(data))]
    random.seed(1024)
    random.shuffle(index)
    print(index)
    dataRandom = data[index]
    labelRandom = label[index]
    delList = np.array([]).astype(int)
    for j in range(16):
        count = 0
        for i in range(len(dataRandom)):
            if (labelRandom[i] == j) & (count < sampleNum):
                delList = np.append(delList, i)
                count = count + 1
    dataRes = np.delete(dataRandom, delList, axis=0)
    labelRes = np.delete(labelRandom, delList, axis=0)
    dataTrain = dataRandom[delList]
    labelTrain = labelRandom[delList]
    return dataTrain, labelTrain, dataRes, labelRes

def data_rd(data, label):
    index = [x for x in range(len(data))]
    random.shuffle(index)
    dataRd = data[index]
    labelRd = label[index]
    return dataRd, labelRd

# 导入数据
preData = scio.loadmat('data/predata/Salinas_corrected.mat')['salinas_corrected'] # 数据
preLabel = scio.loadmat('data/predata/Salinas_gt.mat')['salinas_gt'] # 标签
print('Hyperspectral data shape: ', preData.shape)
print('Label shape: ', preLabel.shape)

# 数据预处理设置
patchSize = 11 # 每个像素点提取 patch 的尺寸
pcaComponents = 30  # 使用 PCA 降维，得到主成分的数量

# PCA降维
# print('\n... ... PCA tranformation ... ...')
# DataPCA = apply_PCA(preData, numComponents=pcaComponents)
# print('Data shape after PCA: ', DataPCA.shape)

# padding
print('\n... ... padding ... ...')
paddingMargin = patchSize // 2
# dataPadding = padding_Zeros(DataPCA, paddingMargin)
dataPadding200 = padding_Zeros(preData, paddingMargin)

# 获取每个像素点的cube
print('\n... ... creating cube... ...')
# dataCubes, dataLabel = get_Cubes(dataPadding, preLabel, paddingMargin, windowSize=patchSize)
dataCubes200, dataLabel200 = get_Cubes(dataPadding200, preLabel, paddingMargin, windowSize=patchSize)

# 释放缓存
del dataPadding200, preLabel, preData
gc.collect()

# 每类取五个样本
print('\n... ... getting samples... ...')
# dataTrain, labelTrain, dataRes, labelRes = get_sample(dataCubes, dataLabel, 5)
dataTrain200, labelTrain200, dataRes200, labelRes200 = get_sample(dataCubes200, dataLabel200, 5)

# 打乱训练数据
# dataTrainRd, labelTrainRd = data_rd(dataTrain, labelTrain)
dataTrainRd200, labelTrainRd200 = data_rd(dataTrain200, labelTrain200)

# 划分候选集与测试集
# dataCdd, dataTest, labelCdd, labelTest = train_test_split(dataRes, labelRes, train_size=0.5, random_state=1024)
dataCdd200, dataTest200, labelCdd200, labelTest200 = train_test_split(dataRes200, labelRes200, train_size=0.5, random_state=1024)


dsName = 'sa3d'
dsName200 = 'sa200'
# 保存处理好的数据
# save_file(dataTrainRd, dsName, sn_train, sn_data, patchSize)
# save_file(labelTrainRd, dsName, sn_train, sn_label, patchSize)
# save_file(dataCdd, dsName, sn_cdd, sn_data, patchSize)
# save_file(labelCdd, dsName, sn_cdd, sn_label, patchSize)
# save_file(dataTest, dsName, sn_test, sn_data, patchSize)
# save_file(labelTest, dsName, sn_test, sn_label, patchSize)
save_file(dataTrainRd200, dsName200, sn_train, sn_data, patchSize)
save_file(labelTrainRd200, dsName200, sn_train, sn_label, patchSize)
save_file(dataCdd200, dsName200, sn_cdd, sn_data, patchSize)
save_file(labelCdd200, dsName200, sn_cdd, sn_label, patchSize)
save_file(dataTest200, dsName200, sn_test, sn_data, patchSize)
save_file(labelTest200, dsName200, sn_test, sn_label, patchSize)



