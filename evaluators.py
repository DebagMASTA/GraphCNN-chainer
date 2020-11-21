#coding:utf-8
import os, time, sys, random
import argparse, yaml, shutil, math
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F

class GraphCnnEvaluator(chainer.training.extensions.Evaluator):
    def __init__(self, iterator, gcnn,
                    converter=chainer.dataset.concat_examples,
                    device=None, eval_hook=None):
        if isinstance(iterator, chainer.dataset.Iterator):
            iterator = {"main":iterator}

        self._iterators = iterator
        self._targets = {"gcnn" : gcnn}
        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook

    def loss_softmax_cross_entropy(self, predict, ground_truth):
        eps = 1e-16
        cross_entropy = -F.mean(F.log(predict + eps) * ground_truth)
        return cross_entropy

    def accuracy(self, predict, ground_truth):
        ground_truth = np.argmax(ground_truth, axis=1)
        ground_truth = ground_truth.astype(np.int32)
        acc = F.accuracy(predict, ground_truth)
        return acc

    def evaluate(self):
        iterator = self._iterators["main"]
        gen = self._targets["gcnn"]

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator) #shallow copy

        summary = chainer.reporter.DictSummary()

        for batch in it:
            observation ={}
            with chainer.reporter.report_scope(observation):
                label, intensity = self.converter(batch, self.device)
                with chainer.using_config("train", False), chainer.using_config('enable_backprop', False):
                        y = gen(intensity)
                loss = self.loss_softmax_cross_entropy(y, label)
                acc = self.accuracy(y, label)

                observation["val1/loss"] = loss
                observation['val1/acc'] = acc

            summary.add(observation)

        return summary.compute_mean()


class GraphCnnEvaluator2(chainer.training.extensions.Evaluator):
    def __init__(self, iterator, gcnn,
                    converter=chainer.dataset.concat_examples,
                    device=None, eval_hook=None):
        if isinstance(iterator, chainer.dataset.Iterator):
            iterator = {"main":iterator}

        self._iterators = iterator
        self._targets = {"gcnn" : gcnn}
        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook

    def loss_softmax_cross_entropy(self, predict, ground_truth):
        eps = 1e-16
        cross_entropy = -F.mean(F.log(predict + eps) * ground_truth)
        return cross_entropy

    def accuracy(self, predict, ground_truth):
        ground_truth = np.argmax(ground_truth, axis=1)
        ground_truth = ground_truth.astype(np.int32)
        acc = F.accuracy(predict, ground_truth)
        return acc

    def evaluate(self):
        iterator = self._iterators["main"]
        gen = self._targets["gcnn"]

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator) #shallow copy

        summary = chainer.reporter.DictSummary()

        for batch in it:
            observation ={}
            with chainer.reporter.report_scope(observation):
                label, intensity = self.converter(batch, self.device)
                with chainer.using_config("train", False), chainer.using_config('enable_backprop', False):
                        y = gen(intensity)
                loss = self.loss_softmax_cross_entropy(y, label)
                acc = self.accuracy(y, label)

                observation["val2/loss"] = loss
                observation['val2/acc'] = acc

            summary.add(observation)

        return summary.compute_mean()
