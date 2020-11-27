# coding:utf-8
import os, random
import numpy as np
import argparse, shutil
import chainer
from chainer import training
from chainer.training import extensions
import utils.ioFunctions as IO
from model import GraphCNN
from dataset import BrainSpectDataset
from updater import GraphCnnUpdater
from evaluators import GraphCnnEvaluator,GraphCnnEvaluator2

def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.backends.cuda.available:
        chainer.backends.cuda.cupy.random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--base', '-B', default=os.path.join(os.path.dirname(os.path.abspath(__file__))),
                        help='base directory path of program files')
    parser.add_argument('--data_dir', '-D', default='D:/PycharmProjects/data/',
                        help='input data directory path')

    parser.add_argument('--out', '-o', default= 'results/test_20201121',
                        help='Directory to output the result')
    parser.add_argument('--flip', '-f', default= True,
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
    train_list='configs/_pn_15964/{}fold/group{}/training_list.txt'.format(args.fold,args.group)
    val1_list='configs/_pn_15964/{}fold/group{}/validation_list.txt'.format(args.fold,args.group)
    val2_list='configs/_pn_15964/{}fold/group{}/test_list.txt'.format(args.fold,args.group)

    if not os.path.exists(os.path.join(args.base, args.out)):
        os.makedirs(os.path.join(args.base, args.out))


    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('')

    print('----- Load dataset -----')
    # Load the dataset
    print('Load adjecency')

    A = np.load(os.path.join(args.data_dir,args.graph))

    train = BrainSpectDataset(data_dir=args.data_dir, data_list_txt=train_list, frip=args.frip, A=A)
    val1 = BrainSpectDataset(data_dir=args.data_dir, data_list_txt=val1_list, frip=False, A=A)
    val2 = BrainSpectDataset(data_dir=args.data_dir, data_list_txt=val2_list, frip=False, A=A)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    val1_iter = chainer.iterators.SerialIterator(val1, args.batchsize,
                                                 repeat=False, shuffle=False)
    val2_iter = chainer.iterators.SerialIterator(val2, args.batchsize,
                                                 repeat=False, shuffle=False)

    print('----- Build model -----')
    gcnn = GraphCNN(train.A,out_dir=os.path.join(args.base, args.out))
    if args.model:
        chainer.serializers.load_npz(args.model, gcnn)

    if args.gpu >= 0:
        chainer.backends.cuda.set_max_workspace_size(2 * 512 * 1024 * 1024)
        chainer.global_config.autotune = True
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        gcnn.to_gpu()  # Copy the model to the GPU

    print('----- Setup optimizer -----')
    optimizer = chainer.optimizers.Adam(alpha=args.alpha)
    optimizer.setup(gcnn)
    optimizer.add_hook(chainer.optimizer.WeightDecay(2e-6))

    print('----- Make updater -----')
    updater = GraphCnnUpdater(
        model = gcnn,
        iterator = train_iter,
        optimizer = {'gcnn':optimizer},
        device = args.gpu
        )

    print('----- Make trainer -----')
    trainer = training.Trainer(updater,
                                (args.epoch, 'epoch'),
                                out=os.path.join(args.base, args.out))
    IO.save_args(os.path.join(args.base, args.out), args)

    snapshot_interval = (args.snapshot_interval, 'epoch')
    display_interval = (args.display_interval, 'epoch')
    evaluation_interval = (args.evaluation_interval, 'epoch')
    trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}.npz'),trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(gcnn, filename='gcnn_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)

    trainer.extend(GraphCnnEvaluator(val1_iter, gcnn, device=args.gpu), trigger=evaluation_interval)
    trainer.extend(GraphCnnEvaluator2(val2_iter, gcnn, device=args.gpu), trigger=evaluation_interval)

    trainer.extend(extensions.LogReport(trigger=display_interval))

    trainer.extend(extensions.ProgressBar(update_interval=10))
    report_keys = ['epoch', 'iteration', 'train/loss','train/acc', 'val1/loss', 'val1/acc','val2/loss', 'val2/acc']
    trainer.extend(extensions.PrintReport(report_keys), trigger=display_interval)

    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(['train/loss', 'val1/loss','val2/loss'], 'epoch', file_name='loss.png', trigger=display_interval))
        trainer.extend(extensions.PlotReport(['train/acc', 'val1/acc','val2/acc'], 'epoch', file_name='accuracy.png', trigger=display_interval))


    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)),'model.py'), '{}/model.py'.format(os.path.join(os.path.dirname(os.path.abspath(__file__)),args.out)))
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)),'dataset.py'), '{}/dataset.py'.format(os.path.join(os.path.dirname(os.path.abspath(__file__)),args.out)))
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)),'training.py'), '{}/training.py'.format(os.path.join(os.path.dirname(os.path.abspath(__file__)),args.out)))

    # Run the training
    print('----- Run the training -----')
    reset_seed(0)
    trainer.run()


if __name__ == '__main__':
    print(1)
    main()
    