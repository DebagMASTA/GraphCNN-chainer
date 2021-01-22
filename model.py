# coding;utf-8
"""
@auther tozawa
"""
import os, sys, time
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F

from utils.functions.pooling.graph_max_pooling import GraphMaxPoolingFunction
from utils.links.connection.graph_convolution import GraphConvolution
from utils.coarsening import coarsen, combine


class SEBlock(chainer.Chain):
    """A squeeze-and-excitation block.
    This block is part of squeeze-and-excitation networks. Channel-wise
    multiplication weights are inferred from and applied to input feature map.
    Please refer to `the original paper
    <https://arxiv.org/pdf/1709.01507.pdf>`_ for more details.
    .. seealso::
        :class:`chainercv.links.model.senet.SEResNet`
    Args:
        n_channel (int): The number of channels of the input and output array.
        ratio (int): Reduction ratio of :obj:`n_channel` to the number of
            hidden layer units.
    """

    def __init__(self, n_channel, ratio=16):
        super(SEBlock, self).__init__()
        reduction_size = n_channel // ratio

        with self.init_scope():
            self.down = L.Linear(n_channel, reduction_size)
            self.up = L.Linear(reduction_size, n_channel)

    def __call__(self, u):
        B, C, N = u.shape

        z = F.average(u, axis=(2))
        x = F.leaky_relu(self.down(z))
        x = F.sigmoid(self.up(x))

        x = F.broadcast_to(x, (N, B, C))
        x = x.transpose((1, 2, 0))

        return u * x


class scSEBlock(chainer.Chain):

    def __init__(self, n_channel, A, ratio=16):
        super(scSEBlock, self).__init__()
        reduction_size = n_channel // ratio

        with self.init_scope():
            self.down = L.Linear(n_channel, reduction_size)
            self.up = L.Linear(reduction_size, n_channel)
            self.sSE = GraphConvolution(in_channels=n_channel, out_channels=1, A=A, K=1)

    def __call__(self, u):
        B, C, N = u.shape

        v = F.sigmoid(self.sSE(u))
        v = v.reshape(B, N)
        v = F.broadcast_to(v, (C, B, N))
        v = v.transpose((1, 0, 2))

        z = F.average(u, axis=(2))
        x = F.leaky_relu(self.down(z))
        x = F.sigmoid(self.up(x))

        x = F.broadcast_to(x, (N, B, C))
        x = x.transpose((1, 2, 0))

        return (u * x) + (u * v)


class BottleNeck(chainer.Chain):
    def __init__(self, n_in, n_mid, n_out, A, use_conv=False):
        w = chainer.initializers.HeNormal()
        super(BottleNeck, self).__init__()
        with self.init_scope():
            self.res1 = GraphConvolution(in_channels=n_in, out_channels=n_mid, A=A, K=1)
            self.bn1 = L.BatchNormalization(n_mid)
            self.res2 = GraphConvolution(in_channels=n_mid, out_channels=n_mid, A=A, K=25)
            self.bn2 = L.BatchNormalization(n_mid)
            self.res3 = GraphConvolution(in_channels=n_mid, out_channels=n_out, A=A, K=1)
            self.bn3 = L.BatchNormalization(n_out)
            if use_conv:
                self.res4 = GraphConvolution(in_channels=n_in, out_channels=n_out, A=A, K=25)
                self.bn4 = L.BatchNormalization(n_out)
        self.use_conv = use_conv

    def __call__(self, x, SE_PRE):
        h = F.leaky_relu(self.bn1(self.res1(x)))
        h = F.leaky_relu(self.bn2(self.res2(h)))
        h = self.bn3(self.res3(h))
        return h + self.bn4(self.res4(SE_PRE)) if self.use_conv else h + SE_PRE


# class ResBlock(chainer.Chain):
#     def __init__(self, n_in, n_out, stride=1, A,K,ksize=1):
#         w = chainer.initializers.HeNormal()
#         super(ResBlock, self).__init__(
#             conv1=GraphConvolution(n_in, n_out, A, K=25),
#             bn1=L.BatchNormalization(n_out),
#             conv2=GraphConvolution(n_in, n_out, A, K=25),
#             bn2=L.BatchNormalization(n_out),
#         )
#     def __call__(self, x, train):
#         h = F.leaky_relu(self.bn1(self.conv1(x), test=not train))
#         h = self.bn2(self.conv2(h), test=not train)
#         if x.data.shape != h.data.shape:
#             xp = chainer.cuda.get_array_module(x.data)
#             n, c, hh, ww = x.data.shape
#             pad_c = h.data.shape[1] - c
#             p = xp.zeros((n, pad_c, hh, ww), dtype=xp.float32)
#             p = chainer.Variable(p, volatile=not train)
#             x = F.concat((p, x))
#             if x.data.shape[2:] != h.data.shape[2:]:
#                 x = F.average_pooling_2d(x, 1, 2)
#         return F.leaky_relu(h + x)


class Block(chainer.ChainList):
    def __init__(self, n_in, n_mid, n_out, n_bottlenecks, A):
        super(Block, self).__init__()
        self.add_link(BottleNeck(n_in, n_mid, n_out, A, True))
        for _ in range(n_bottlenecks - 1):
            self.add_link(BottleNeck(n_out, n_mid, n_out))

    def __call__(self, x, SE_PRE):
        for f in self:
            x = f(x, SE_PRE)
        return x


class GraphCNN(chainer.Chain):
    def __init__(self, A, in_channels=5, out_channels=4, **kwargs):
        """
        This model is based on https://github.com/maggie0106/Graph-CNN-in-3D-Point-Cloud-Classification
        """
        initializer = chainer.initializers.HeNormal()
        super(GraphCNN, self).__init__()

        if kwargs['out_dir'] is not None:

            # Precompute the coarsened graphs
            graphs, pooling_inds = coarsen(A, levels=6)
            # In order to simulate 2x2 max pooling, combine the 4 levels
            # of graphs into 2 levels by combining pooling indices.
            graphs, pooling_inds = combine(graphs, pooling_inds, 2)

            np.save(kwargs['out_dir'] + '/graphs.npy', graphs)
            np.save(kwargs['out_dir'] + '/pooling_inds.npy', pooling_inds)

        else:
            graphs = np.load(kwargs['graph_dir'] + 'graphs.npy', allow_pickle=True)
            pooling_inds = np.load(kwargs['graph_dir'] + 'pooling_inds.npy', allow_pickle=True)

        with self.init_scope():
            self.gc1 = GraphConvolution(in_channels=in_channels, out_channels=16, A=graphs[0], K=25)
            self.bn1 = L.BatchNormalization(16)
            self.p1 = GraphMaxPoolingFunction(pooling_inds[0])

            self.se1_1 = scSEBlock(16, A=graphs[1])
            self.r1_1 = Block(16, 16, 64, 1, A=graphs[1])
            self.se1_2 = scSEBlock(64, A=graphs[1])
            self.r1_2 = Block(64, 16, 64, 1, A=graphs[1])
            self.se1_3 = scSEBlock(64, A=graphs[1])
            self.r1_3 = Block(64, 16, 64, 1, A=graphs[1])
            self.p2 = GraphMaxPoolingFunction(pooling_inds[1])

            self.se2_1 = scSEBlock(64, A=graphs[2])
            self.r2_1 = Block(64, 32, 128, 1, A=graphs[2])
            self.se2_2 = scSEBlock(128, A=graphs[2])
            self.r2_2 = Block(128, 32, 128, 1, A=graphs[2])
            self.se2_3 = scSEBlock(128, A=graphs[2])
            self.r2_3 = Block(128, 32, 128, 1, A=graphs[2])
            self.se2_4 = scSEBlock(128, A=graphs[2])
            self.r2_4 = Block(128, 32, 128, 1, A=graphs[2])
            self.p3 = GraphMaxPoolingFunction(pooling_inds[2])
            # self.se3_1 = SEBlock(256, 16)
            # self.r3_1 = Block(256, 128, 512, 1, A=graphs[3])
            # self.se3_2 = SEBlock(512, 16)
            # self.r3_2 = Block(512, 128, 512, 1, A=graphs[3])
            # self.se3_3 = SEBlock(512, 16)
            # self.r3_3 = Block(512, 128, 512, 1, A=graphs[3])
            # self.se3_4 = SEBlock(512, 16)
            # self.r3_4 = Block(512, 128, 512, 1, A=graphs[3])
            # self.se3_5 = SEBlock(512, 16)
            # self.r3_5 = Block(512, 128, 512, 1, A=graphs[3])
            # self.se3_6 = SEBlock(512, 16)
            # self.r3_6 = Block(512, 128, 512, 1, A=graphs[3])
            # self.p4 = GraphMaxPoolingFunction(pooling_inds[3])
            #
            # self.se4_1=SEBlock(512,16)
            # self.r4_1 = Block(512, 256, 1024, 1, A=graphs[4])
            # self.se4_2=SEBlock(1024,16)
            # self.r4_2 = Block(1024, 256, 1024, 1, A=graphs[4])
            # self.se4_3=SEBlock(1024,16)
            # self.r4_3 = Block(1024, 256, 1024, 1, A=graphs[4])

            self.fc1 = L.Linear(None, 512, initialW=initializer)
            self.bnfc1 = L.BatchNormalization(512)
            self.fc2 = L.Linear(None, out_channels, initialW=initializer)
            self.bnfc2 = L.BatchNormalization(out_channels)

    def forward(self, x):
        """
        x: shape = (batchsize, in_channels, N)
        """

        h = F.leaky_relu(self.bn1(self.gc1(x)))
        # print("o1_1:",o1_1.shape)
        # del e0
        h = self.p1(h)
        h = F.leaky_relu(self.r1_1(self.se1_1(h), h))
        h = F.leaky_relu(self.r1_2(self.se1_2(h), h))
        # h=F.leaky_relu(self.r1_3(h,h))
        h = self.p2(h)

        h = F.leaky_relu(self.r2_1(self.se2_1(h), h))
        h = F.leaky_relu(self.r2_2(self.se2_2(h), h))
        # h=F.leaky_relu(self.r2_3(h,h))
        # h=F.leaky_relu(self.r2_4(h,h))
        h = self.p3(h)

        # o3_1=F.leaky_relu(self.r3_1(self.se3_1(o2),o2))
        # o3_2=F.leaky_relu(self.r3_2(self.se3_2(o3_1),o3_1))
        # o3_3=F.leaky_relu(self.r3_3(self.se3_3(o3_2),o3_2))
        # o3_4=F.leaky_relu(self.r3_4(self.se3_4(o3_3),o3_3))
        # o3_5=F.leaky_relu(self.r3_5(self.se3_5(o3_4),o3_4))
        # o3_6=F.leaky_relu(self.r3_6(self.se3_6(o3_5),o3_5))
        # o3 = self.p4(o3_6)
        #
        # o4_1 = F.leaky_relu(self.r4_1(self.se4_1(o3),o3))
        # o4_2=F.leaky_relu(self.r4_2(self.se4_2(o4_1),o4_1))
        # o4_3=F.leaky_relu(self.r4_3(self.se4_3(o4_2),o4_2))
        dropout_ratio = 0.5

        h = F.leaky_relu(F.dropout(self.bnfc1(self.fc1(h)), dropout_ratio))
        h = F.softmax(self.bnfc2(self.fc2(h)), axis=1)

        return h