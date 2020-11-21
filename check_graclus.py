from utils.coarsening import coarsen, combine
import numpy as np
import os
import argparse, shutil
import chainer.functions as F
from chainer import Variable
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int, default=0,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--base', '-B', default=os.path.join(os.path.dirname(os.path.abspath(__file__))),
                    help='base directory path of program files')
parser.add_argument('--data_dir', '-D', default='D:/PycharmProjects/data/',
                    help='input data directory path')

parser.add_argument('--out', '-o', default= 'results/test',
                    help='Directory to output the result')
parser.add_argument('--frip', '-f', default= True,
                    help='horizontal frip or not')
parser.add_argument('--graph', '-gr', default="Graph/NN8.npy",
                    help='graph directory path')

# training configs
parser.add_argument('--batchsize', '-b', type=int, default=16,
                    help='Number of images in each mini-batch')
parser.add_argument('--epoch', type=int, default=50,
                    help='Number of epoch')

parser.add_argument('--snapshot_interval', type=int, default=1)
parser.add_argument('--display_interval', type=int, default=1)
parser.add_argument('--evaluation_interval', type=int, default=1)
parser.add_argument('--config_path','-C', type=str, default='configs/parameters.yml',
                    help='path to config file')

parser.add_argument('--model', '-m', default=
# os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/0715_9/3fold/group1/gcnn_iter_50.npz')
                    '',
                    help='Load model data')
parser.add_argument('--resume', '-res', default=
# os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/0715_9/3fold/group1/snapshot_iter_50.npz')
                    '',
                    help='Resume the training from snapshot')

parser.add_argument('--fold', '-FO',  type=int, default=3)
parser.add_argument('--group', '-GR',  type=int, default=1)

parser.add_argument('--alpha', default=0.0002,
                    help='alpha of Adam')

args = parser.parse_args()

# ab=np.arange(5)
# aaa=np.arange(5)
# bs=4
# bb=np.full(5,5)
# for i in range(bs):
#     ab=ab+bb
#     aaa=np.append(aaa,ab)
# print("aaa",aaa)


# A = np.load(os.path.join(args.data_dir, args.graph))
#
# graphs, pooling_inds = coarsen(A, levels=6)
#
# print(pooling_inds)
# graphs, pooling_inds = combine(graphs, pooling_inds, 2)
# print(pooling_inds[2],type(pooling_inds[0]))


# x=np.arange(12)
# print(x)
#
# x=np.reshape(x,[6,2])
# print(x,'\n\n\n')
#
# x=np.reshape(x,[3,4])
# print(x,'\n\n\n')
#
# x=np.reshape(x,[2,3,2])
# print(x,'\n\n\n')
#
# x=np.reshape(x,[2,6])
# print(x)

# dict={}
# for i in range(9):
#     a=np.arange(i)
#     print(a)
#     dict['cluster{}'.format(i)]=a
#
# print('dict',dict)
# j=2
# print('AAA',dict['cluster{}'.format(j)])
#
# x=y=1
# print(x,y)


def loss_softmax_cross_entropy( predict, ground_truth):
    eps = 1e-16
    cross_entropy = -F.mean(F.log(predict + eps) * ground_truth)
    return cross_entropy



def MyCrossEntropyLoss(predict, ground_truth):
    eps = 1e-16
    ground_truth = torch.nn.functional.one_hot(ground_truth, num_classes=4)
    print('predict\n',predict)
    print('ground_truth\n',ground_truth)

    cross_entropy = -torch.mean(torch.log(predict + eps) * ground_truth)

    return cross_entropy


# a=np.arange(4)*0.1
# a=np.vstack([a,a])
# a=Variable(a)
# print(a)
#
# gt=np.array([[0,1,0,0],[0,0,1,0]]).astype(np.float64)
# gt=Variable(gt)
#
# print(gt)
#
# loss=loss_softmax_cross_entropy(a,gt)
#
# print(loss)


b=np.arange(4)*0.1
b=np.vstack([b,b])
b=torch.from_numpy(b.astype(np.float32)).clone()

print(b)

gt=np.array([1,2]).astype(np.int64)

gt=torch.from_numpy(gt.astype(np.int64)).clone()

loss=MyCrossEntropyLoss(b,gt)

print(loss)

