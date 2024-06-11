import numpy as np
from select_sample import *
from train_test import *
from torch.utils.data import DataLoader
from config import *
from core_set import *
from Entropy import EntropySampling
from GCNAL import *
from BADGE import init_centers
import argparse
import sys

device = torch.device("cuda:0")

class Logger(object):
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wetherTrain", type=float, default=1,
                        help="Whether the model is pre trained")
    parser.add_argument("-me", "--method", type=str, default='P_GCN_AL',
                        help="method choosing")
    parser.add_argument("-m", "--model", type=str, default='IPmodel/model_train_CYF_150_15.pth',
                        help="Model choosing")
    parser.add_argument("-n", "--modelName", type=str, default='CYF',
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

    # 记载训练集数据进行预训练
    if args.wetherTrain == 0:

        if args.modelName == 'HybridSN':
            trainDs = data_set3D(trainData, trainLabel)
            model = HybridSN(numClass=9, patchSize=19).to(device)
            pretrain_model(model, trainDs, epoch=150, learningRate=0.001, modelName=args.modelName, patchsize=19)
            exit()
        elif args.modelName == 'SSRN':
            trainDs = data_set3D(trainData, trainLabel)
            model = SSRN(numClass=16, patchSize=9).to(device)
            pretrain_model(model, trainDs, epoch=150, learningRate=0.001, modelName=args.modelName, patchsize=9)
            exit()
        elif args.modelName == '3DCNN':
            trainDs = data_set3D(trainData, trainLabel)
            model = Net_3DCNN(numClass=9).to(device)
            pretrain_model(model, trainDs, epoch=200, learningRate=0.001, modelName=args.modelName, patchsize=19)
            exit()
        elif args.modelName == 'CYF':
            trainDs = data_set3D(trainData, trainLabel)
            model = Net_Proposed(patch_size=19).to(device)
            pretrain_model(model, trainDs, epoch=150, learningRate=0.002, modelName=args.modelName, patchsize=19)
            exit()
        elif args.modelName == 'A2K':
            trainDs = data_set3D(trainData, trainLabel)
            model = S3KAIResNet(band=103, classes=9, reduction=2).to(device)
            pretrain_model(model, trainDs, epoch=150, learningRate=0.002, modelName=args.modelName, patchsize=11)
            exit()


    # 导入预训练后的模型
    modelAfterPt = torch.load(args.model)

    # 导入候选集数据
    candidateData = np.load(args.cddData)
    candidateLabel = np.load(args.cddLabel)

    # 导入测试数据
    testData = np.load(args.testData)
    testLabel = np.load(args.testLabel)
    testSet = data_set3D(testData, testLabel)

    Cepoch = 50
    modelEpoch = 150
    patchsize = 19
    if args.method == 'P_GCN_AL':
        testnum = 'dis90_IP'

        file = open('P_GCN_AL+'+str(args.modelName)+'_'+str(modelEpoch)+'_'+str(patchsize)+'_'+testnum+'.txt', 'w')
        sys.stdout = Logger('P_GCN_AL+'+str(args.modelName)+'_'+str(modelEpoch)+'_'+str(patchsize)+'_'+testnum+'.txt')

        # 测试结果
        model_test(testSet, modelAfterPt)

        # 挑选样本并继续训练网络
        for cycle in range(CYCLE):

            # 加载已标记数据，提取特征
            labeledData = trainData
            labeledLabel = trainLabel
            trainSet = data_set3D(labeledData, labeledLabel)
            trainLoader = DataLoader(trainSet, batch_size=16)
            labeledFeats = get_features(modelAfterPt, trainLoader)

            saveF = labeledFeats.detach().cpu().numpy()
            np.save('labeledF'+str(cycle)+'.npy', saveF)
            np.save('Labeled_L'+str(cycle)+'.npy', labeledLabel)

            # 加载候选集数据
            CddData = candidateData
            CddLabel = candidateLabel
            CddSet = data_set3D(CddData, CddLabel)
            CddLoader = DataLoader(CddSet, batch_size=16)
            CddFeats = get_features(modelAfterPt, CddLoader)

            featureDim = CddFeats.shape[1]

            # 找到特征聚合后的原型
            protoClassNum = np.max(labeledLabel) + 1
            protoClassNum = protoClassNum.astype(int)
            #
            # # 找到每一类的原型zA
            protoTypeIndex = prototypeFinding(labeledFeats.detach().cpu().numpy(), labeledLabel, protoClassNum)
            protoTypeFeats = labeledFeats[protoTypeIndex]
            np.save('prototype'+str(cycle)+'.npy', protoTypeIndex)

            # 利用图卷积提取特征
            GCNEpoch = [50, 50, 50, 50, 50]
            GCNselectSample = GCNselectsamples(protoTypeFeats, CddFeats, protoClassNum, featsDim=featureDim, GCNepoch=GCNEpoch[cycle])


            # 将挑选的数据加入训练集
            trainData = np.concatenate((labeledData, CddData[GCNselectSample]), axis=0)
            trainLabel = np.concatenate((labeledLabel, CddLabel[GCNselectSample]), axis=0)
            trainConSet = data_set3D(trainData, trainLabel)

            # 将挑选的数据从候选集删除
            candidateData = np.delete(CddData, GCNselectSample, axis=0)
            candidateLabel = np.delete(CddLabel, GCNselectSample, axis=0)

            # 使用扩充后的训练集继续模型
            modelAfterPt = train_Continue(modelAfterPt, trainConSet, Cepoch=Cepoch, testDs=testSet, learningRate=0.001, cycle=cycle, modelName=args.modelName)

        save_net(modelAfterPt, 'res_SA2' + str(args.method), patchsize=19)


    elif args.method == 'random':
        testnum = 'hy_IP'
        file = open('log/random+'+str(args.modelName)+'_'+str(modelEpoch)+'_'+str(patchsize)+'_'+str(testnum)+'.txt', 'w')
        sys.stdout = Logger(file.name)

        # 测试结果
        model_test(testSet, modelAfterPt)

        for cycle in range(CYCLE):

            # 加载训练集及候选集数据
            labeledData = trainData
            labeledLabel = trainLabel
            CddData = candidateData
            CddLabel = candidateLabel

            # 随机采样
            addData, addLabel, resData, resLabel = randomMethod(CddData, CddLabel)

            # 将挑选的数据加入训练集
            trainData = np.concatenate((labeledData, addData), axis=0)
            trainLabel = np.concatenate((labeledLabel, addLabel), axis=0)
            trainConSet = data_set3D(trainData, trainLabel)

            # 将挑选的数据从候选集删除
            candidateData = resData
            candidateLabel = resLabel

            # 使用扩充后的训练集继续模型
            modelAfterPt = train_Continue(modelAfterPt, trainConSet, Cepoch=Cepoch, testDs=testSet, learningRate=0.001, cycle=cycle, modelName=args.modelName)

        save_net(modelAfterPt, 'res_IP' + str(args.method), patchsize=15)

    elif args.method == 'Coreset':
        testnum = 'hy1_PU'
        file = open('log/Coreset+'+str(args.modelName)+'_'+str(modelEpoch)+'_'+str(patchsize)+'_'+str(testnum)+'.txt', 'w')
        sys.stdout = Logger(file.name)

        # 测试结果
        model_test(testSet, modelAfterPt)

        for cycle in range(CYCLE):

            # 加载训练集及候选集数据
            labeledData = trainData
            labeledLabel = trainLabel
            labeledDS = data_set3D(labeledData, labeledLabel)
            labeledLoader = DataLoader(labeledDS, batch_size=16)
            CddData = candidateData
            CddLabel = candidateLabel
            unlabeledDS = data_set3D(CddData, CddLabel)
            unlabeledLoader = DataLoader(unlabeledDS, batch_size=16)

            # sample
            classNum = np.max(labeledLabel) + 1
            sampleSize = int(classNum * 2)
            sample_rows = active_sample(labeledLoader, unlabeledLoader, sampleSize, model=modelAfterPt)

            # 将挑选的数据加入训练集
            trainData = np.concatenate((labeledData, CddData[sample_rows]), axis=0)
            trainLabel = np.concatenate((labeledLabel, CddLabel[sample_rows]), axis=0)
            trainConSet = data_set3D(trainData, trainLabel)

            # 将挑选的数据从候选集删除
            candidateData = np.delete(CddData, sample_rows, axis=0)
            candidateLabel = np.delete(CddLabel, sample_rows, axis=0)

            # 使用扩充后的训练集继续模型
            modelAfterPt = train_Continue(modelAfterPt, trainConSet, Cepoch=Cepoch, testDs=testSet, learningRate=0.001, cycle=cycle, modelName=args.modelName)

        save_net(modelAfterPt, 'res_PU_' + str(args.method), patchsize=19)

    elif args.method == 'Entropy':
        testnum = 'hy1_PU'
        file = open('log/Entropy+'+str(args.modelName)+'_'+str(modelEpoch)+'_'+str(patchsize)+'_'+str(testnum)+'.txt', 'w')
        sys.stdout = Logger(file.name)

        # 测试结果
        model_test(testSet, modelAfterPt)

        for cycle in range(CYCLE):


            CddData = candidateData
            CddLabel = candidateLabel
            unlabeledDS = data_set3D(CddData, CddLabel)
            unlabeledLoader = DataLoader(unlabeledDS, batch_size=16)

            classNum = (np.max(CddLabel) + 1).astype(int)

            sampleSelectIdx = EntropySampling(modelAfterPt, unlabeledLoader, classNum)

            # 将挑选的数据加入训练集
            trainData = np.concatenate((trainData, CddData[sampleSelectIdx]), axis=0)
            trainLabel = np.concatenate((trainLabel, CddLabel[sampleSelectIdx]), axis=0)
            trainConSet = data_set3D(trainData, trainLabel)

            # 将挑选的数据从候选集删除
            candidateData = np.delete(CddData, sampleSelectIdx, axis=0)
            candidateLabel = np.delete(CddLabel, sampleSelectIdx, axis=0)

            # 使用扩充后的训练集继续模型
            modelAfterPt = train_Continue(modelAfterPt, trainConSet, Cepoch=Cepoch, testDs=testSet, learningRate=0.001, cycle=cycle, modelName=args.modelName)

        save_net(modelAfterPt, 'res_PU'+str(args.method), patchsize=19)

    elif args.method == 'BADGE':
        testnum = 'hy1_PU'
        file = open('log/BADGE+'+str(args.modelName)+'_'+str(modelEpoch)+'_'+str(patchsize)+'_'+str(testnum)+'.txt', 'w')
        sys.stdout = Logger(file.name)

        # 测试结果
        model_test(testSet, modelAfterPt)

        for cycle in range(CYCLE):

            CddData = candidateData
            CddLabel = candidateLabel
            unlabeledDS = data_set3D(CddData, CddLabel)
            unlabeledLoader = DataLoader(unlabeledDS, batch_size=16)

            classNum = (np.max(CddLabel) + 1).astype(int)
            samplesize = (classNum * 2).astype(int)

            gradEmbedding = get_features(modelAfterPt, unlabeledLoader)
            gradEmbedding = gradEmbedding.detach().cpu().numpy()
            sampleSelectIdx = init_centers(gradEmbedding, samplesize)

            # 将挑选的数据加入训练集
            trainData = np.concatenate((trainData, CddData[sampleSelectIdx]), axis=0)
            trainLabel = np.concatenate((trainLabel, CddLabel[sampleSelectIdx]), axis=0)
            trainConSet = data_set3D(trainData, trainLabel)

            # 将挑选的数据从候选集删除
            candidateData = np.delete(CddData, sampleSelectIdx, axis=0)
            candidateLabel = np.delete(CddLabel, sampleSelectIdx, axis=0)

            # 使用扩充后的训练集继续模型
            modelAfterPt = train_Continue(modelAfterPt, trainConSet, Cepoch=Cepoch, testDs=testSet, learningRate=0.001, cycle=cycle, modelName=args.modelName)

        save_net(modelAfterPt, 'res_PU' + str(args.method), patchsize=19)

    elif args.method == 'UNGCN':
        testnum = 'hy1_SA'
        file = open('log/UNGCN+' + str(args.modelName) + '_' + str(modelEpoch) + '_' + str(patchsize) + '_' + str(
            testnum) + '.txt', 'w')
        sys.stdout = Logger(file.name)

        # 测试结果
        model_test(testSet, modelAfterPt)

        for cycle in range(CYCLE):
            # 加载已标记数据，提取特征
            labeledData = trainData
            labeledLabel = trainLabel
            trainSet = data_set3D(labeledData, labeledLabel)
            trainLoader = DataLoader(trainSet, batch_size=16)
            labeledFeats = get_features(modelAfterPt, trainLoader)

            # 加载候选集数据
            CddData = candidateData
            CddLabel = candidateLabel
            CddSet = data_set3D(CddData, CddLabel)
            CddLoader = DataLoader(CddSet, batch_size=16)
            CddFeats = get_features(modelAfterPt, CddLoader)

            classNum = (np.max(CddLabel) + 1).astype(int)
            samplesize = (classNum * 2).astype(int)

            sampleSelectIdx = UncertainGCN(CddFeats, labeledFeats, samplesize)

            # 将挑选的数据加入训练集
            trainData = np.concatenate((trainData, CddData[sampleSelectIdx]), axis=0)
            trainLabel = np.concatenate((trainLabel, CddLabel[sampleSelectIdx]), axis=0)
            trainConSet = data_set3D(trainData, trainLabel)

            # 将挑选的数据从候选集删除
            candidateData = np.delete(CddData, sampleSelectIdx, axis=0)
            candidateLabel = np.delete(CddLabel, sampleSelectIdx, axis=0)

            # 使用扩充后的训练集继续模型
            modelAfterPt = train_Continue(modelAfterPt, trainConSet, Cepoch=Cepoch, testDs=testSet, learningRate=0.001, cycle=cycle,
                                          modelName=args.modelName)

        save_net(modelAfterPt, 'res_SA' + str(args.method), patchsize=19)

    elif args.method == 'COREGCN':
        testnum = 'hy1_SA'
        file = open('log/COREGCN+' + str(args.modelName) + '_' + str(modelEpoch) + '_' + str(patchsize) + '_' + str(
            testnum) + '.txt', 'w')
        sys.stdout = Logger(file.name)

        # 测试结果
        model_test(testSet, modelAfterPt)

        for cycle in range(CYCLE):
            # 加载已标记数据，提取特征
            labeledData = trainData
            labeledLabel = trainLabel
            trainSet = data_set3D(labeledData, labeledLabel)
            trainLoader = DataLoader(trainSet, batch_size=16)
            labeledFeats = get_features(modelAfterPt, trainLoader)

            # 加载候选集数据
            CddData = candidateData
            CddLabel = candidateLabel
            CddSet = data_set3D(CddData, CddLabel)
            CddLoader = DataLoader(CddSet, batch_size=16)
            CddFeats = get_features(modelAfterPt, CddLoader)

            classNum = (np.max(CddLabel) + 1).astype(int)
            samplesize = (classNum * 2).astype(int)

            sampleSelectIdx = CoreGCN(CddFeats, labeledFeats, samplesize)

            # 将挑选的数据加入训练集
            trainData = np.concatenate((trainData, CddData[sampleSelectIdx]), axis=0)
            trainLabel = np.concatenate((trainLabel, CddLabel[sampleSelectIdx]), axis=0)
            trainConSet = data_set3D(trainData, trainLabel)

            # 将挑选的数据从候选集删除
            candidateData = np.delete(CddData, sampleSelectIdx, axis=0)
            candidateLabel = np.delete(CddLabel, sampleSelectIdx, axis=0)

            # 使用扩充后的训练集继续模型
            modelAfterPt = train_Continue(modelAfterPt, trainConSet, Cepoch=Cepoch, testDs=testSet, learningRate=0.001, cycle=cycle,
                                          modelName=args.modelName)

        save_net(modelAfterPt, 'res_SA' + str(args.method), patchsize=15)