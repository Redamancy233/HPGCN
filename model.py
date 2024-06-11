import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import nn
import argparse

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.linear1 = nn.Linear(nhid, nhid)
        self.linear2 = nn.Linear(nhid, 2)


    def forward(self, x, adj):
        x = F.leaky_relu(self.gc1(x, adj))
        x = F.leaky_relu(self.gc2(x, adj))
        # x = self.linear1(x)
        result2 = self.linear2(x)
        return x, result2


class HybridSN(nn.Module):

    def __init__(self, numClass, patchSize):
        super(HybridSN, self).__init__()
        # 3DConv
        self.conv3d_1 = nn.Sequential(
        nn.Conv3d(1, 8, kernel_size=(7, 3, 3), stride=1, padding=0),
        nn.BatchNorm3d(8),
        nn.ReLU(inplace=True),)
        self.conv3d_2 = nn.Sequential(
        nn.Conv3d(8, 16, kernel_size=(5, 3, 3), stride=1, padding=0),
        nn.BatchNorm3d(16),
        nn.ReLU(inplace=True),)
        self.conv3d_3 = nn.Sequential(
        nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=0),
        nn.BatchNorm3d(32),
        nn.ReLU(inplace=True))

      # 2DConv
        self.conv2d_4 = nn.Sequential(
            nn.Conv2d(576, 64, kernel_size=(3, 3), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Linear((64 * (patchSize - 4 * 2) ** 2), 256) #18496
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, numClass)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        out = self.conv3d_1(x)
        out = self.conv3d_2(out)
        out = self.conv3d_3(out)
        out = self.conv2d_4(out.reshape(out.shape[0], -1, out.shape[3], out.shape[4]))
        out = out.reshape(out.shape[0], -1)
        out = F.relu(self.dropout(self.fc1(out)))
        out1 = F.relu(self.dropout(self.fc2(out)))
        out = self.fc3(out1)
        return out, out1

# SSRN
class SPCModule(nn.Module):
    def __init__(self, in_channels, out_channels, patchSize, bias=True):
        super(SPCModule, self).__init__()

        self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(patchSize, 1, 1), stride=(2, 1, 1), bias=False)

    def forward(self, input):

        out = self.s1(input)

        return out


class SPAModule(nn.Module):
    def __init__(self, in_channels, out_channels, k, bias=True):
        super(SPAModule, self).__init__()

        # print('k=',k)
        self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(k, 3, 3), bias=False)
        # self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        out = self.s1(input)
        out = out.squeeze(2)
        return out


class ResSPC(nn.Module):
    def __init__(self, in_channels, out_channels, patchSize, bias=True):
        super(ResSPC, self).__init__()

        self.spc1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(patchSize, 1, 1), padding=(patchSize//2, 0, 0), bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(in_channels), )

        self.spc2 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(patchSize, 1, 1), padding=(patchSize//2, 0, 0), bias=False),
            nn.LeakyReLU(inplace=True), )

        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, input):
        out = self.spc1(input)
        out = self.bn2(self.spc2(out))

        return F.leaky_relu(out + input)


class ResSPA(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(ResSPA, self).__init__()

        self.spa1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                                  nn.LeakyReLU(inplace=True),
                                  nn.BatchNorm2d(in_channels), )

        self.spa2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                  nn.LeakyReLU(inplace=True), )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        out = self.spa1(input)
        out = self.bn2(self.spa2(out))

        return F.leaky_relu(out + input)


class SSRN(nn.Module):
    def __init__(self, numClass, patchSize):
        super(SSRN, self).__init__()

        self.layer1 = SPCModule(1, 28, patchSize)
        # self.bn1 = nn.BatchNorm3d(28)

        self.layer2 = ResSPC(28, 28, patchSize)

        self.layer3 = ResSPC(28, 28, patchSize)

        # self.layer4 = SPAModule(28, 28, k=(100-patchSize // 2))
        self.layer4 = SPAModule(28, 28, k=(100 - patchSize // 2))
        self.bn4 = nn.BatchNorm2d(28)

        self.layer5 = ResSPA(28, 28)
        self.layer6 = ResSPA(28, 28)

        self.fc = nn.Linear(28, numClass)

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))  # self.bn1(F.leaky_relu(self.layer1(x)))
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn4(F.leaky_relu(self.layer4(x)))
        x = self.layer5(x)
        x = self.layer6(x)

        x = F.avg_pool2d(x, x.size()[-1])
        x = self.fc(x.squeeze())
        x1 = x

        return x, x1


# 3D-CNN
class Net_3DCNN(nn.Module):
    """
    DEEP FEATURE EXTRACTION AND CLASSIFICATION OF HYPERSPECTRAL IMAGES BASED ON
                        CONVOLUTIONAL NEURAL NETWORKS
    Yushi Chen, Hanlu Jiang, Chunyang Li, Xiuping Jia and Pedram Ghamisi
    IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2017
    """
    def __init__(self,  numClass):
        super(Net_3DCNN, self).__init__()

        self.conv1 = nn.Conv3d(1, 8, (3, 3, 3))
        self.pool1 = nn.MaxPool3d((2, 2, 2))
        self.conv2 = nn.Conv3d(8, 16, (3, 3, 3))
        self.pool2 = nn.MaxPool3d((2, 2, 2))
        self.fc = nn.Linear(432, numClass) #384IP #432PU
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x1 = self.fc(x)
        return x1, x


# CYF
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return self.relu(x)


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=2):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        # self.SEweight=SEWeightModule(channels=inp)
    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        # seweights=self.SEweight(x)
        out = identity * a_w * a_h

        return out

class conv2d_Resblock(nn.Module):
    def __init__(self, in_channels):
        super(conv2d_Resblock, self).__init__()
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
        )
        self.Coordattention = CoordAtt(in_channels // 4, in_channels // 4)
        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out_conv2d_1 = self.conv2d_1(x)
        out_att = self.Coordattention(out_conv2d_1)
        out_conv2d_2 = self.conv2d_2(out_att)
        out = x + out_conv2d_2
        return out


class Net_Proposed(nn.Module):
    def __init__(self, patch_size):
        super(Net_Proposed, self).__init__()
        self.patchsize = patch_size
        self.Conv3d_base_1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True)
        )

        self.Conv3d_base_2 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )

        self.Conv3d_base_3 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )


        self.conv2d_10 = nn.Sequential(
            nn.Conv2d(144, 128, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.CAt1 = CoordAtt(inp=128, oup=128)
        self.conv2d_base_1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2d_block1 = conv2d_Resblock(64)


        self.conv2d_20 = nn.Sequential(
            nn.Conv2d(320, 128, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.CAt2 = CoordAtt(inp=128, oup=128)
        self.conv2d_base_2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2d_block2 = conv2d_Resblock(64)

        self.conv2d_30 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.CAt3 = CoordAtt(inp=128, oup=128)
        self.conv2d_base_3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2d_block3 = conv2d_Resblock(64)

        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc3 = nn.Linear(192, 9)

    def forward(self, x):
        out_scale1 = self.Conv3d_base_1(x)

        out_scale2 = self.Conv3d_base_2(x)

        out_scale3 = self.Conv3d_base_3(x)

        out_2dscale1 = out_scale1.reshape(out_scale1.shape[0], -1, self.patchsize - 2, self.patchsize - 2)
        out_2dscale1=self.conv2d_10(out_2dscale1)
        out_att1 = self.CAt1(out_2dscale1)
        out_2dscale1 = self.conv2d_base_1(out_att1)
        out_2dscale1 = self.conv2d_block1(out_2dscale1)

        out_2dscale2 = out_scale2.reshape(out_scale2.shape[0], -1, self.patchsize - 4, self.patchsize - 4)
        out_2dscale2 = torch.cat((out_2dscale1, out_2dscale2), 1)
        out_2dscale2=self.conv2d_20(out_2dscale2)
        out_att2 = self.CAt2(out_2dscale2)
        out_2dscale2 = self.conv2d_base_2(out_att2)
        out_2dscale2 = self.conv2d_block2(out_2dscale2)

        out_2dscale3 = out_scale3.reshape(out_scale3.shape[0], -1, self.patchsize - 6, self.patchsize - 6)
        out_2dscale3 = torch.cat((out_2dscale2, out_2dscale3), 1)
        out_2dscale3=self.conv2d_30(out_2dscale3)
        out_att3 = self.CAt3(out_2dscale3)
        out_2dscale3 = self.conv2d_base_3(out_att3)
        out_2dscale3 = self.conv2d_block3(out_2dscale3)

        out1 = self.avgpool1(out_2dscale1)
        out1 = out1.reshape(out1.shape[0], -1)

        out2 = self.avgpool2(out_2dscale2)
        out2 = out2.reshape(out2.shape[0], -1)

        out3 = self.avgpool3(out_2dscale3)
        out3 = out3.reshape(out3.shape[0], -1)

        # out=out1+out2+out3
        out = torch.cat((out1, out2, out3), dim=1)
        out4 = self.fc3(out)
        return out4, out

parser = argparse.ArgumentParser(description='Training for HSI')
parser.add_argument(
    '-k',
    '--kernel',
    type=int,
    dest='kernel',
    default=24,
    help="Length of kernel")
args = parser.parse_args()
PARAM_KERNEL_SIZE = args.kernel

class ChannelSELayer3D(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.relu(
            self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(
            input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        return output_tensor
class SpatialSELayer3D(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor,
                                  squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor
class ChannelSpatialSELayer3D(nn.Module):
    """
       3D extension of concurrent spatial and channel squeeze & excitation:
           *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
       """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(
            self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor
class ProjectExciteLayer(nn.Module):
    """
        Project & Excite Module, specifically designed for 3D inputs
        *quote*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ProjectExciteLayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.relu = nn.ReLU()
        self.conv_c = nn.Conv3d(
            in_channels=num_channels,
            out_channels=num_channels_reduced,
            kernel_size=1,
            stride=1)
        self.conv_cT = nn.Conv3d(
            in_channels=num_channels_reduced,
            out_channels=num_channels,
            kernel_size=1,
            stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()

        # Average along channels and different axes
        squeeze_tensor_w = F.adaptive_avg_pool3d(input_tensor, (1, 1, W))

        squeeze_tensor_h = F.adaptive_avg_pool3d(input_tensor, (1, H, 1))

        squeeze_tensor_d = F.adaptive_avg_pool3d(input_tensor, (D, 1, 1))

        # tile tensors to original size and add:
        final_squeeze_tensor = sum([
            squeeze_tensor_w.view(batch_size, num_channels, 1, 1, W),
            squeeze_tensor_h.view(batch_size, num_channels, 1, H, 1),
            squeeze_tensor_d.view(batch_size, num_channels, D, 1, 1)
        ])

        # Excitation:
        final_squeeze_tensor = self.sigmoid(
            self.conv_cT(self.relu(self.conv_c(final_squeeze_tensor))))
        output_tensor = torch.mul(input_tensor, final_squeeze_tensor)

        return output_tensor
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv2d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w, t = x.size()

        # feature descriptor on the global spatial information
        # 24, 1, 1, 1
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -3)).transpose(
            -1, -3).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
class Residual(nn.Module):  # pytorch
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding,
            use_1x1conv=False,
            stride=1,
            start_block=False,
            end_block=False,
    ):
        super(Residual, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride), nn.ReLU())
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride)
        if use_1x1conv:
            self.conv3 = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

        if not start_block:
            self.bn0 = nn.BatchNorm3d(in_channels)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

        if start_block:
            self.bn2 = nn.BatchNorm3d(out_channels)

        if end_block:
            self.bn2 = nn.BatchNorm3d(out_channels)

        # ECA Attention Layer
        self.ecalayer = eca_layer(out_channels)

        # start and end block initialization
        self.start_block = start_block
        self.end_block = end_block

    def forward(self, X):
        identity = X

        if self.start_block:
            out = self.conv1(X)
        else:
            out = self.bn0(X)
            out = F.relu(out)
            out = self.conv1(out)

        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)

        if self.start_block:
            out = self.bn2(out)

        out = self.ecalayer(out)

        out += identity

        if self.end_block:
            out = self.bn2(out)
            out = F.relu(out)

        return out
class S3KAIResNet(nn.Module):
    def __init__(self, band, classes, reduction):
        super(S3KAIResNet, self).__init__()
        self.name = 'SSRN'
        self.conv1x1 = nn.Conv3d(
            in_channels=1,
            out_channels=PARAM_KERNEL_SIZE,
            kernel_size=(1, 1, 7),
            stride=(1, 1, 2),
            padding=0)
        self.conv3x3 = nn.Conv3d(
            in_channels=1,
            out_channels=PARAM_KERNEL_SIZE,
            kernel_size=(3, 3, 7),
            stride=(1, 1, 2),
            padding=(1, 1, 0))

        self.batch_norm1x1 = nn.Sequential(
            nn.BatchNorm3d(
                PARAM_KERNEL_SIZE, eps=0.001, momentum=0.1,
                affine=True),  # 0.1
            nn.ReLU(inplace=True))
        self.batch_norm3x3 = nn.Sequential(
            nn.BatchNorm3d(
                PARAM_KERNEL_SIZE, eps=0.001, momentum=0.1,
                affine=True),  # 0.1
            nn.ReLU(inplace=True))

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv_se = nn.Sequential(
            nn.Conv3d(
                PARAM_KERNEL_SIZE, band // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True))
        self.conv_ex = nn.Conv3d(
            band // reduction, PARAM_KERNEL_SIZE, 1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.res_net1 = Residual(
            PARAM_KERNEL_SIZE,
            PARAM_KERNEL_SIZE, (1, 1, 7), (0, 0, 3),
            start_block=True)
        self.res_net2 = Residual(PARAM_KERNEL_SIZE, PARAM_KERNEL_SIZE,
                                 (1, 1, 7), (0, 0, 3))
        self.res_net3 = Residual(PARAM_KERNEL_SIZE, PARAM_KERNEL_SIZE,
                                 (3, 3, 1), (1, 1, 0))
        self.res_net4 = Residual(
            PARAM_KERNEL_SIZE,
            PARAM_KERNEL_SIZE, (3, 3, 1), (1, 1, 0),
            end_block=True)

        kernel_3d = math.ceil((band - 10) / 2)
        # print(kernel_3d)

        self.conv2 = nn.Conv3d(
            in_channels=PARAM_KERNEL_SIZE,
            out_channels=128,
            padding=(0, 0, 0),
            kernel_size=(1, 1, 3),
            stride=(1, 1, 1))
        self.batch_norm2 = nn.Sequential(
            nn.BatchNorm3d(128, eps=0.001, momentum=0.1, affine=True),  # 0.1
            nn.ReLU(inplace=True))
        self.conv3 = nn.Conv3d(
            in_channels=1,
            out_channels=PARAM_KERNEL_SIZE,
            padding=(0, 0, 0),
            kernel_size=(3, 3, 128),
            stride=(1, 1, 1))
        self.batch_norm3 = nn.Sequential(
            nn.BatchNorm3d(
                PARAM_KERNEL_SIZE, eps=0.001, momentum=0.1,
                affine=True),  # 0.1
            nn.ReLU(inplace=True))

        self.avg_pooling = nn.AvgPool3d(kernel_size=(5, 5, 1))
        self.full_connection = nn.Sequential(
            nn.Linear(480, classes)
            # nn.Softmax()
        )

    def forward(self, X):
        x_1x1 = self.conv1x1(X)
        x_1x1 = self.batch_norm1x1(x_1x1).unsqueeze(dim=1)
        x_3x3 = self.conv3x3(X)
        x_3x3 = self.batch_norm3x3(x_3x3).unsqueeze(dim=1)

        x1 = torch.cat([x_3x3, x_1x1], dim=1)
        U = torch.sum(x1, dim=1)
        S = self.pool(U)
        Z = self.conv_se(S)
        attention_vector = torch.cat(
            [
                self.conv_ex(Z).unsqueeze(dim=1),
                self.conv_ex(Z).unsqueeze(dim=1)
            ],
            dim=1)
        attention_vector = self.softmax(attention_vector)
        V = (x1 * attention_vector).sum(dim=1)

        x2 = self.res_net1(V)
        x2 = self.res_net2(x2)
        x2 = self.batch_norm2(self.conv2(x2))
        x2 = x2.permute(0, 4, 2, 3, 1)
        x2 = self.batch_norm3(self.conv3(x2))

        x3 = self.res_net3(x2)
        x3 = self.res_net4(x3)
        x4 = self.avg_pooling(x3)
        x4 = x4.view(x4.size(0), -1)
        return self.full_connection(x4), x4