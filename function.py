from Dataset import *
from sklearn.metrics.pairwise import cosine_similarity

# 数据文件保存
def save_file(data_file, ds_name, divide, dorl, patchsize):
    np.save('./data/'+str(patchsize)+'/'+ds_name+'_'+divide+'_'+dorl+'_'+str(patchsize)+'.npy', data_file)

# 网络保存
def save_net(model, model_name, patchsize):
    torch.save(model, 'model_trained/model_'+model_name+'_'+str(patchsize)+'.pth')

# 化为邻接矩阵
def aff_to_adj(x):
    x = x.detach().cpu().numpy()
    # adjcos = cosine_similarity(x, x)
    adj = np.matmul(x, x.transpose())
    adj += -1.0*np.eye(adj.shape[0])
    adj_diag = np.sum(adj, axis=0) #rowise sum
    adj = np.matmul(adj, np.diag(1/adj_diag))
    adj = adj + np.eye(adj.shape[0])
    adj = torch.Tensor(adj).cuda()
    # adj = torch.Tensor(adjcos).cuda()
    return adj

# 提取特征
def get_features(model, data_loader):
    model.eval()
    features = torch.tensor([]).cuda()
    with torch.no_grad():
        for inputs,  _ in data_loader:
            inputs = inputs.cuda()
            output, features_batch = model(inputs)
            features = torch.cat((features, features_batch), 0)
        feat = features
    return feat


