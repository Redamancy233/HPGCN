import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from Dataset import data_set3D
import torch

def get_Cubes(data, label, margin, windowSize, removeZeroLabels = True):

    # split patches
    patchesData = np.zeros((data.shape[0] * data.shape[1], windowSize, windowSize, data.shape[2]))
    patchesLabels = np.zeros((data.shape[0] * data.shape[1]))

    # 获取cubes
    patchesIndex = 0
    for r in range(margin, data.shape[0] - margin):
        for c in range(margin, data.shape[1] - margin):
            patch = data[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchesIndex, :, :, :] = patch
            patchesLabels[patchesIndex] = label[r-margin, c-margin]
            patchesIndex = patchesIndex + 1

    # 去0标签
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1

    return patchesData, patchesLabels

def padding_Zeros(data, margin):
    newData = np.zeros((data.shape[0] + 2 * margin, data.shape[1] + 2 * margin, data.shape[2]))
    xOffset = margin
    yOffset = margin
    newData[xOffset:data.shape[0] + xOffset, yOffset:data.shape[1] + yOffset, :] = data
    return newData

def apply_PCA(data, numComponents):

    # PCA需要二维形式，对数据变形
    newData = np.reshape(data, (-1, data.shape[2]))

    # PCA降维
    pca = PCA(n_components=numComponents, whiten=True)
    dataPCA = pca.fit_transform(newData)

    # 将数据还原成三维
    newDataPCA = np.reshape(dataPCA, (data.shape[0], data.shape[1], numComponents))

    return newDataPCA
def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi,
                        ground_truth.shape[0] * 2.0 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)
    return 0
def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 5:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 6:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 7:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 8:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 9:
            y[index] = np.array([128, 128, 0]) / 255.
        if item == 10:
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 11:
            y[index] = np.array([128, 0, 128]) / 255.
        if item == 12:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 13:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 14:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 15:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 16:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 17:
            y[index] = np.array([215, 255, 0]) / 255.
        if item == 18:
            y[index] = np.array([0, 255, 215]) / 255.
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255.
    return y
def generate_png(all_iter, net, gt_hsi, device, path):
    pred_test = []
    net.eval()
    for X, y in all_iter:
        X = X.to(device)
        output, _ = net(X)
        pred_test.extend(output.cpu().argmax(axis=1).detach().numpy())
    outputs = np.zeros((height, width))
    k = 0
    for i in range(height):
        for j in range(width):
            if int(preLabel[i, j]) == 0:
                continue
            else:
                outputs[i][j] = pred_test[k] + 1
                k = k + 1
    gt = gt_hsi.flatten()
    outputs=outputs.flatten()
    for i in range(len(gt)):
        if gt[i] == 0:
            gt[i] = 17
            outputs[i] = 17
    gt = gt[:] - 1
    outputs=outputs[:]-1

    x = np.ravel(outputs)
    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)
    y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    classification_map(y_re, gt_hsi, 300,
                       path + 'predict_UNGCN.png')
    classification_map(gt_re, gt_hsi, 300,
                       path + 'groundtruth.png')
    print('------Get classification maps successful-------')

# # 导入数据
# preData = scio.loadmat('data/predata/PaviaU.mat')['paviaU'] # 数据
# preLabel = scio.loadmat('data/predata/PaviaU_gt.mat')['paviaU_gt'] # 标签
# print('Hyperspectral data shape: ', preData.shape)
# print('Label shape: ', preLabel.shape)
#
# height = preLabel.shape[0]
# width = preLabel.shape[1]
#
# # 数据预处理设置
# patchSize = 19  # 每个像素点提取 patch 的尺寸
# pcaComponents = 20  # 使用 PCA 降维，得到主成分的数量
#
# # PCA降维
# print('\n... ... PCA tranformation ... ...')
# DataPCA = apply_PCA(preData, numComponents=pcaComponents)
# print('Data shape after PCA: ', DataPCA.shape)
#
# # padding
# print('\n... ... padding ... ...')
# paddingMargin = patchSize // 2
# dataPadding = padding_Zeros(DataPCA, paddingMargin)
#
# # 获取每个像素点的cube
# print('\n... ... creating cube... ...')
# dataCubes, dataLabel = get_Cubes(dataPadding, preLabel, paddingMargin, windowSize=patchSize)
#
# np.save('data/hy/PU/data.npy', dataCubes)
# np.save('data/hy/PU/label.npy', dataLabel)

# hy
preLabel = scio.loadmat('data/predata/PaviaU_gt.mat')['paviaU_gt'] # 标签
dataHy = np.load('data/hy/PU/data.npy')
labelHy = np.load('data/hy/PU/label.npy')

huanyuan_set = data_set3D(dataHy, labelHy)
huanyuan_loader = DataLoader(dataset=huanyuan_set, batch_size=277, shuffle=False)

height = preLabel.shape[0]
width = preLabel.shape[1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Model = torch.load('hy/PU/model_res_PUUNGCN_19.pth')
generate_png(huanyuan_loader, Model, preLabel,  device, 'fig/PU/hy/')