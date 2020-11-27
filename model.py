# coding;utf-8
"""
@auther tozawa
"""
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from utils.functions.pooling.graph_max_pooling import GraphMaxPoolingFunction
from utils.links.connection.graph_convolution import GraphConvolution
from utils.coarsening import coarsen, combine

#with training, you need to give out_dir. predict, give graph_dir(out_dir of training)
class GraphCNN(chainer.Chain):
    def __init__(self, A,  in_channels=5, out_channels=4,**kwargs):
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

            np.save(kwargs['out_dir']+'/graphs.npy',graphs)
            np.save(kwargs['out_dir']+'/pooling_inds.npy',pooling_inds)

        else:
            graphs=np.load(kwargs['graph_dir']+'graphs.npy', allow_pickle=True)
            pooling_inds=np.load(kwargs['graph_dir']+'pooling_inds.npy', allow_pickle=True)

        with self.init_scope():
            self.gc1 = GraphConvolution(in_channels=in_channels, out_channels=32, A=graphs[0], K=25)
            self.p1 = GraphMaxPoolingFunction(pooling_inds[0])
            self.gc2 = GraphConvolution(in_channels=None, out_channels=64, A=graphs[1], K=25)
            self.p2 = GraphMaxPoolingFunction(pooling_inds[1])
            self.gc3 = GraphConvolution(in_channels=None, out_channels=64, A=graphs[2], K=25)
            self.p3 = GraphMaxPoolingFunction(pooling_inds[2])

            self.fc1 = L.Linear(None, 512, initialW=initializer)
            self.fc2 = L.Linear(None, out_channels, initialW=initializer)

    def forward(self, x,return_featuremap=False):
        """
        x: shape = (batchsize, in_channels, N)
        """
        # y=x.data
        # print("y:",y,y.shape)
        # # y=chainer.backends.cuda.to_cpu(x.data)
        # # y=chainer.cuda.to_cpu(y)
        # y=np.array(y)
        # print(y.shape)
        # y=np.reshape(y,[16][5][15964])
        # y=y[0][0]
        # y=pd.DataFrame(y)
        # y.to_csv('D:/PycharmProjects/GraphCNN-for-Brain-Spect/results/0714/test.csv')

        dropout_ratio = 0.5
        # h = F.relu(self.gc1(x))

        h = self.p1(F.relu(self.gc1(x)))
        h = self.p2(F.relu(self.gc2(h)))
        h = self.p3(F.relu(self.gc3(h)))
        h_fc1 = F.relu(F.dropout(self.fc1(h), dropout_ratio))

        if return_featuremap==True:
            return h_fc1
        else:
            h = F.softmax(self.fc2(h_fc1),axis=1)

            return h