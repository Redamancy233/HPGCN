from select_sample import *
from train_test import *
from torch.utils.data import DataLoader
from config import *
from GCNAL import *
import argparse

device = torch.device("cuda:0")

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wetherTrain", type=float, default=1,
                        help="Whether the model is pre trained")
    parser.add_argument("-m", "--model", type=str, default='IPmodel/model_train_HybridSN_150_15.pth',
                        help="Model choosing")
    parser.add_argument("-n", "--modelName", type=str, default='3DCNN',
                        help="Model name")
    parser.add_argument("-a", "--trainData", type=str, default='data/15(rs128)/ip3d_train_data_15.npy',
                        help="data of the training set")
    parser.add_argument("-al", "--trainLabel", type=str, default='data/15(rs128)/ip3d_train_label_15.npy',
                        help="data of the training set")
    parser.add_argument("-c", "--cddData", type=str, default='data/15(rs128)/ip3d_cdd_data_15.npy',
                        help="data of the candidate set")
    parser.add_argument("-cl", "--cddLabel", type=str, default='data/15(rs128)/ip3d_cdd_label_15.npy',
                        help="label of the candidate set")
    parser.add_argument("-e", "--testData", type=str, default='data/15(rs128)/ip3d_test_data_15.npy',
                        help="data of the testing set")
    parser.add_argument("-el", "--testLabel", type=str, default='data/15(rs128)/ip3d_test_label_15.npy',
                        help="label of the testing set")
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parser()

    # 导入训练数据
    trainData = np.load(args.trainData)
    trainLabel = np.load(args.trainLabel)
    # trainDs = data_set3D(trainData, trainLabel)
    # model = Net_3DCNN(numClass=9).to(device)
    # pretrain_model(model, trainDs, epoch=200, learningRate=0.001, modelName=args.modelName, patchsize=19)

    # # 导入预训练后的模型
    modelAfterPt = torch.load(args.model)

    # 导入候选集数据
    candidateData = np.load(args.cddData)
    candidateLabel = np.load(args.cddLabel)

    # 导入测试数据
    testData = np.load(args.testData)
    testLabel = np.load(args.testLabel)
    testSet = data_set3D(testData, testLabel)

    # 首次测试
    model_test(testSet, modelAfterPt)

    Cepoch = 50
    modelEpoch = 200
    patchsize = 11
    for cycle in range(CYCLE):
        # 加载已标记数据，提取特征
        labeledData = trainData
        labeledLabel = trainLabel
        trainSet = data_set3D(labeledData, labeledLabel)
        trainLoader = DataLoader(trainSet, batch_size=16)
        labeledFeats = get_features(modelAfterPt, trainLoader)
        labelFeats = labeledFeats.detach().cpu().numpy()

        # 加载候选集数据
        CddData = candidateData
        CddLabel = candidateLabel
        CddSet = data_set3D(CddData, CddLabel)
        CddLoader = DataLoader(CddSet, batch_size=16)
        CddFeats = get_features(modelAfterPt, CddLoader)

        # featureDim = CddFeats.shape[1]

        # 原型数量
        protoClassNum = np.max(labeledLabel) + 1
        protoClassNum = protoClassNum.astype(int)

        # 找到每一类的原型zA
        protoTypeIndex = prototypeFinding(labelFeats, labeledLabel, protoClassNum)
        protoTypeFeats = labelFeats[protoTypeIndex]
        CddFeats = CddFeats.detach().cpu().numpy()

        Distance = euclidean_distances(CddFeats, protoTypeFeats)
        protoDisIndex = []
        for j in range(len(CddFeats)):
            disIndex = np.argmin(Distance[j])
            protoDisIndex.append(disIndex)

        proClassnum = len(np.unique(protoDisIndex))
        proClassIndex = np.unique(protoDisIndex)

        selectIndex = []
        for z in range(proClassnum):
            protoClassIndex = np.where(protoDisIndex == proClassIndex[z])[0]

            Prototype = np.expand_dims(protoTypeFeats[z], axis=0)
            protoDistance = euclidean_distances(Prototype, CddFeats[protoClassIndex])
            maxDistance = protoClassIndex[np.argmax(protoDistance)]
            minDistance = protoClassIndex[np.argmin(protoDistance)]
            selectIndex.append(maxDistance)
            selectIndex.append(minDistance)

        if proClassnum < protoClassNum:
            unfindpoint = []
            IDX = [undix for undix in range(protoClassNum)]
            unfind = np.setdiff1d(IDX, proClassIndex)
            for uf in range(len(unfind)):
                iter = unfind[uf]
                UFDis = Distance[iter]
                UFSort = np.argsort(UFDis)
                unfindpoint.append(UFSort[0])
                unfindpoint.append(UFSort[1])
            selectIndex = np.concatenate((selectIndex, unfindpoint))

        # 将挑选的数据加入训练集
        trainData = np.concatenate((labeledData, CddData[selectIndex]), axis=0)
        trainLabel = np.concatenate((labeledLabel, CddLabel[selectIndex]), axis=0)
        trainConSet = data_set3D(trainData, trainLabel)

        # 将挑选的数据从候选集删除
        candidateData = np.delete(CddData, selectIndex, axis=0)
        candidateLabel = np.delete(CddLabel, selectIndex, axis=0)

        # 使用扩充后的训练集继续模型
        modelAfterPt = train_Continue(modelAfterPt, trainConSet, Cepoch=Cepoch, testDs=testSet, learningRate=0.001,
                                      cycle=cycle, modelName=args.modelName)

