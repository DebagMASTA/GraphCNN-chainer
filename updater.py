#coding:utf-8

import os, time, sys
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
import pandas as pd

class GraphCnnUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gcnn = kwargs.pop('model')
        super(GraphCnnUpdater, self).__init__(*args, **kwargs)

    def loss_softmax_cross_entropy(self, predict, ground_truth):
        eps = 1e-16

        cross_entropy = -F.mean(F.log(predict + eps) * ground_truth)
        return cross_entropy

    def accuracy(self, predict, ground_truth):
        ground_truth = np.argmax(ground_truth, axis=1)
        ground_truth = ground_truth.astype(np.int32)
        acc = F.accuracy(predict, ground_truth)
        return acc

    def update_core(self):
        optimizer = self.get_optimizer('gcnn')

        batch = self.get_iterator('main').next()
        label, intensity = self.converter(batch, self.device)

        gcnn = self.gcnn

        y = gcnn(intensity)
        # x=chainer.backends.cuda.to_cpu(x.data)
        # # # y=chainer.cuda.to_cpu(y)
        # x=np.array(x)
        # print(x.shape)
        # x=np.reshape(x,[16][5][15964])
        # x=x[0][0]
        # x=pd.DataFrame(x)
        # x.to_csv('D:/PycharmProjects/GraphCNN-for-Brain-Spect/results/0714/test.csv')

        loss = self.loss_softmax_cross_entropy(y,label)
        acc=self.accuracy(y,label)
        gcnn.cleargrads()
        loss.backward()
        optimizer.update()
        chainer.reporter.report({'train/loss':loss, 'train/acc':acc})


