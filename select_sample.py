import numpy as np
from function import *
from model import *
from sklearn.metrics.pairwise import euclidean_distances
import torch.optim as optim
import gc
from config import *
import random


device = torch.device("cuda:0")



# 寻找原型
def prototypeFinding(feats, labels, classNum):
    LFeats = feats
    LLablels = labels
    # # 提取后的特征为tensor，需转为np
    # LFeats = LFeats.detach().cpu().numpy()
    ProtoTypes = np.zeros(classNum).astype(int)
    for i in range(classNum):
        ClassIndex = []
        for j in range(len(LFeats)):
            if LLablels[j] == i:
                ClassIndex.append(j)
        ClassFeats = LFeats[ClassIndex]
        # 求每一类各样本间的距离
        ClassDistance = euclidean_distances(ClassFeats, ClassFeats)
        ClassDisMean = np.mean(ClassDistance, axis=1)
        ClassDisMin = np.argmin(ClassDisMean)
        ProtoTypes[i] = ClassIndex[ClassDisMin]
    return ProtoTypes

def GCNselectsamples(LabeledFeat, CddFeat, classNum, featsDim, GCNepoch):
    # 将候选集数据特征与原型数据特征进行拼接
    DataFeats = torch.cat((CddFeat, LabeledFeat), 0)
    # 候选集赋予标签0
    label0 = torch.zeros(len(CddFeat), dtype=int)
    # 原型赋予标签1
    label1 = torch.ones(len(LabeledFeat), dtype=int)
    # 将候选集伪标签与原型伪标签进行拼接
    labelGCN = torch.cat((label0, label1), 0)
    # 计算候选集数据特征的邻接矩阵
    adj = aff_to_adj(DataFeats)

    # 加载GCN模型
    modelGCN = GCN(featsDim, featsDim).to(device)
    # 加载优化器、loss函数相关参数
    optimizer = optim.Adam(modelGCN.parameters(), lr=1e-2, weight_decay=5e-3)
    Loss = nn.CrossEntropyLoss()

    lossData = []

    # GCN模型训练
    print('GCN training begin')
    for i in range(GCNepoch):
        optimizer.zero_grad()
        DataGCN = DataFeats.to(device)
        LabelGCN = labelGCN.to(device)
        _, outputs = modelGCN(DataGCN, adj)
        loss = Loss(outputs, LabelGCN)
        loss.backward()
        optimizer.step()
        lossData.append(loss.item())
    print('GCN training finished')
    # np.save('lossGCN_SA'+str(GCNepoch)+'.npy', lossData)
    # 获取关系特征
    modelGCN.eval()
    inputs = DataFeats.cuda()
    GCNGetFeats, _ = modelGCN(inputs, adj)
    CddGCNFeat = GCNGetFeats[:len(CddFeat)].detach().cpu().numpy()
    ProtoFeat = GCNGetFeats[len(CddFeat):].detach().cpu().numpy()
    Distance = euclidean_distances(CddGCNFeat, ProtoFeat)

    # 依据距离计算候选集样本距离哪个原型最近
    protoDisIndex = []
    for j in range(len(CddGCNFeat)):
        disIndex = np.argmin(Distance[j])
        protoDisIndex.append(disIndex)

    proClassnum = len(np.unique(protoDisIndex))
    proClassIndex = np.unique(protoDisIndex)

    # 找出每一类中离每类原型最近、最远的点
    # PrototypeFeat = LabeledFeat.detach().cpu().numpy()

    selectIndex = []
    for z in range(proClassnum):
        protoClassIndex = np.where(protoDisIndex == proClassIndex[z])[0]

        Prototype = np.expand_dims(ProtoFeat[z], axis=0)
        protoDistance = euclidean_distances(Prototype, CddGCNFeat[protoClassIndex])
        maxDistance = protoClassIndex[np.argmax(protoDistance)]
        minDistance = protoClassIndex[np.argmin(protoDistance)]
        selectIndex.append(maxDistance)
        selectIndex.append(minDistance)

    # Newcenter = np.zeros(proClassnum).astype(int)
    # # 找出该类未标记样本的中心点
    # for z in range(proClassnum):
    #     ClassIndex = np.where(protoDisIndex == proClassIndex[z])[0]
    #     UlClassFeats = CddGCNFeat[ClassIndex]
    #     # 求每一类各样本间的距离
    #     ulClassDistance = euclidean_distances(UlClassFeats, UlClassFeats)
    #     ulClassDisMean = np.mean(ulClassDistance, axis=1)
    #     ulClassDisMin = np.argmin(ulClassDisMean)
    #     Newcenter[z] = ClassIndex[ulClassDisMin]
    #
    # # 找出离该类未标记样本的最远点
    # maxIdx = []
    # for mc in range(proClassnum):
    #     nowUl = np.where(protoDisIndex == proClassIndex[mc])[0]
    #     center = np.expand_dims(CddGCNFeat[Newcenter[mc]], axis=0)
    #     maxNowDis = euclidean_distances(center, CddGCNFeat[nowUl])
    #     maxidx = np.argmax(maxNowDis)
    #     maxIdx.append(nowUl[maxidx])
    #
    # selectIndex = np.concatenate((Newcenter, maxIdx))
    if proClassnum < classNum:
        unfindpoint = []
        IDX = [undix for undix in range(classNum)]
        unfind = np.setdiff1d(IDX, proClassIndex)
        for uf in range(len(unfind)):
            iter = unfind[uf]
            UFDis = Distance[iter]
            UFSort = np.argsort(UFDis)
            unfindpoint.append(UFSort[0])
            unfindpoint.append(UFSort[1])
        selectIndex = np.concatenate((selectIndex, unfindpoint))

    print(len(selectIndex))
    del adj, modelGCN, CddGCNFeat, ProtoFeat, DataFeats, label1, label0
    torch.cuda.empty_cache()
    gc.collect()

    return selectIndex

# random
def randomMethod(data, label):
    index = [x for x in range(len(data))]
    random.shuffle(index)
    dataRd = data[index]
    labelRd = label[index]
    classNum = np.max(labelRd) + 1
    addNum = (classNum * 2).astype(int)
    dataSelect = dataRd[:addNum]
    labelSelect = labelRd[:addNum]
    dataRes = dataRd[addNum:]
    labelRes = labelRd[addNum:]
    return dataSelect, labelSelect, dataRes, labelRes


