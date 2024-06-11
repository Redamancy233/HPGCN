import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader
from model import *
from function import *

device = torch.device("cuda:0")

#预训练
def pretrain_model(model, train_ds, epoch, learningRate, modelName, patchsize):

    # 初始化损失函数、优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate)

    total_loss = 0
    y_train = train_ds.label
    train_dl = DataLoader(train_ds, batch_size=40)
    trainloss = []
    acc_all = []

    # 训练
    for epoch in range(epoch):
        train_pred = []
        for i, (data, label) in enumerate(train_dl):
            input = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs, _ = model(input)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            train_pred = np.concatenate((train_pred, outputs))

        # 显示每一周期的loss、训练准确率
        acc_c = accuracy_score(y_train, train_pred, normalize=True)
        acc_all.append(acc_c)
        trainloss.append(loss.item())
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (epoch + 1, total_loss / (epoch + 1), loss.item()))
        print('[Train acc: %4f]' % acc_c)

    # 保存loss数据
    with open("loss/train_loss_"+str(modelName)+"_"+str(epoch+1)+".txt", 'w') as train_loss:
        train_loss.write(str(trainloss))

    # 保存网络
    save_net(model, 'train_'+str(modelName)+'_'+str(epoch+1), patchsize=patchsize)

# 继续训练
def train_Continue(model, trainSet, Cepoch, testDs, learningRate, cycle, modelName):

    # if modelName == 'HybridSN':
    #     model = HybridSN(numClass=16, patchSize=15).to(device)
    #
    # elif modelName == 'SSRN':
    #     model = SSRN(numClass=16, patchSize=15).to(device)


    # 加载训练好的模型
    modelContinue = model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modelContinue.parameters(), lr=learningRate)

    total_loss = 0
    accBest = 0
    epochBest = 0
    kappaBest = 0
    train_dl = DataLoader(trainSet, batch_size=16)

    # 继续训练网络
    for epoch in range(Cepoch):
        train_pred = []
        for i, (data, label) in enumerate(train_dl):
            input = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs, _ = modelContinue(input)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            train_pred = np.concatenate((train_pred, outputs))
        # print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (epoch + 1, total_loss / (epoch + 1), loss.item()))
        # print(epoch+1)

    # 测试
    modelContinue.eval()
    testDl = DataLoader(testDs, batch_size=16, shuffle=False)
    labelTest = testDs.label
    testPred = []

    with torch.no_grad():
        for data, _ in testDl:
            input = data.to(device)
            outputs, _ = modelContinue(input)
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            testPred = np.concatenate((testPred, outputs))

        # acc_s = accuracy_score(labelTest, testPred, normalize=True)
        acc_c = classification_report(labelTest, testPred, digits=4)
        confusionM = confusion_matrix(labelTest, testPred)
        kappa = cohen_kappa_score(labelTest, testPred)
        # if (acc_s > accBest) & (epoch > 0):
        #     epochBest = epoch
        #     accBest = acc_s
        #     kappaBest = kappa
        #     acc_c = classification_report(labelTest, testPred, digits=4)
        #     confusionM = confusion_matrix(labelTest, testPred)
        #     accCBest = acc_c
        #     confusionMB = confusionM
        #     modelBest = modelContinue


    print('cycle:', (cycle + 1))
    print(acc_c)
    print(confusionM)
    print(kappa)
    # print('Best Acc:', accBest)
    # print('Best kappa:', kappaBest)
    # print('Best epoch:', (epochBest + 1))
    # print(accCBest)
    # print(confusionMB)
    return modelContinue

def model_test(testDs, model):
    model.eval()
    testDl = DataLoader(testDs, batch_size=25)
    labelTest = testDs.label
    testPred = []

    for data, _ in testDl:
        input = data.to(device)
        outputs, _ = model(input)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        testPred = np.concatenate((testPred, outputs))

    acc_c = classification_report(labelTest, testPred, digits=4)
    kappa = cohen_kappa_score(labelTest, testPred)
    confusionM = confusion_matrix(labelTest, testPred)
    print(acc_c)
    print(kappa)
    print(confusionM)


