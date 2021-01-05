#coding:utf-8
'''
* @auther tzw
* @date 2018-7-10
'''

import os, sys, time
import argparse, yaml, shutil, math
import numpy as np
import chainer
import csv
import pandas as pd

from model import GraphCNN
import chainer.functions as F
from dataset import BrainSpectDataset
import matplotlib.pyplot as plt
import time
import argparse
import json
import utils.yaml_utils  as yaml_utils
import re


def main():
    filename="/accuracy.csv"
    matrix_filename= "/confusion_matrix.csv"

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                        help='base directory path of program files')
    # parser.add_argument('--graph', '-gp', default="results/generated_graph/NN8.npy",
    #                     help='horizontal frip or not')
    parser.add_argument('--input', default='results/test/',
                        help='Path to log file')
    parser.add_argument('--loss_minimam', default=False,
                        help='how to select epoch ')
    args = parser.parse_args()
    num_of_label=4
    config = yaml_utils.Config(yaml.load(open(args.input+'config.yml')))
    data_dir=config.data_dir
    fold=config.fold
    group=config.group

    test1_list='configs/_pn_15964/{}fold/group{}/test_list.txt'.format(fold,group)
    test2_list='configs/_pn_15964/{}fold/group{}/validation_list.txt'.format(fold,group)
    graph_path=config.graph
    out1_dir=os.path.join(args.base, args.input)+'prediction_val1'
    out2_dir=os.path.join(args.base, args.input)+'prediction_val2'


    print('GPU: {}'.format(args.gpu))
    print('')

    log_path=args.input+'log'
    filename="/accuracy.csv"
    matrix_filename= "/confusion_matrix.csv"
    acc_curve_name="/accuracy_curve_test.png"
    loss_curve_name="/loss_curve_test.png"

    if not os.path.exists(out1_dir):
        os.makedirs(out1_dir)
    if not os.path.exists(out2_dir):
        os.makedirs(out2_dir)

    num = 100 / 100
    # Load JSON
    json_dict = json.load(open(log_path, 'r'))
    name_list = ["train/loss","train/acc", "epoch", "iteration", "elapsed_time", "val1/loss", "val2/loss", "val1/acc", "val2/acc"]
    epoch_list = np.zeros(int(len(json_dict) * num))
    itr_list = np.zeros(int(len(json_dict) * num))
    train_loss_list = np.zeros(int(len(json_dict) * num))
    val1_loss_list = np.zeros(int(len(json_dict) * num))
    val2_loss_list = np.zeros(int(len(json_dict) * num))
    train_acc_list= np.zeros(int(len(json_dict) * num))
    val1_acc_list = np.zeros(int(len(json_dict) * num))
    val2_acc_list = np.zeros(int(len(json_dict) * num))
    loss = np.zeros([3, int(len(json_dict) * num)])
    loss_temp = np.zeros([3, int(len(json_dict) * num)])

    acc = np.zeros([3, int(len(json_dict) * num)])

    """
    epoch_list = [0]*len(json_dict)
    itr_list = [0]*len(json_dict)
    loss_list = [0]*len(json_dict)
    val_loss_list = [0]*len(json_dict)
    """
    # labelnumA = 3
    # labelnumP = 3
    # Dice_list_A = np.zeros([labelnumA, int(len(json_dict) * num)])
    # val_Dice_list_A = np.zeros([labelnumA, int(len(json_dict) * num)])
    # Dice_list_P = np.zeros([labelnumP, int(len(json_dict) * num)])
    # val_Dice_list_P = np.zeros([labelnumP, int(len(json_dict) * num)])

    for name in range(int(len(json_dict) * num)):
        epoch_list[name] = json_dict[name]["epoch"]
        itr_list[name] = json_dict[name]["iteration"]
        train_loss_list[name] = json_dict[name]["train/loss"]
        val1_loss_list[name] = json_dict[name]["val1/loss"]
        val2_loss_list[name] = json_dict[name]["val2/loss"]
        train_acc_list[name] = json_dict[name]["train/acc"]
        val1_acc_list[name] = json_dict[name]["val1/acc"]
        val2_acc_list[name] = json_dict[name]["val2/acc"]

    # calc ave of train/loss and val/loss
    loss[0] = train_loss_list
    loss[1] = val1_loss_list
    loss[2] = val2_loss_list

    w = np.array([1, 1])
    # loss_ave = np.average(loss, weights = w, axis = 0)

    acc[0] = train_acc_list
    acc[1] = val1_acc_list
    acc[2] = val2_acc_list

    def loss_optimal(loss_2,acc_2):
        values = []
        indexes = []
        for i in range(len(loss_2)):
            if loss_2[i]==min(loss_2):
                values.append(acc_2[i])
                indexes.append(i)
        optimal_index=indexes[np.argmax(values)]
        return(optimal_index)

    def acc_optimal(loss_3,acc_3):
        values = []
        indexes = []
        for i in range(len(loss_3)):
            if acc_3[i]==max(acc_3):
                values.append(loss_3[i])
                indexes.append(i)
        optimal_index=indexes[np.argmin(values)]
        return(optimal_index)

    if args.loss_minimam==True:
        val1_optimal_epoch =loss_optimal(loss[1],acc[1])
        val2_optimal_epoch=loss_optimal(loss[2],acc[2])

    if args.loss_minimam==False:
        val1_optimal_epoch=acc_optimal(loss[1],acc[1])
        val2_optimal_epoch=acc_optimal(loss[2],acc[2])


    val1_acc_max_value = np.max(acc[1])
    val2_acc_max_value = np.max(acc[2])
    val1_loss_min_value=np.min(loss[1])
    val2_loss_min_value=np.min(loss[2])
    print(val1_optimal_epoch)
    print(epoch_list[val1_optimal_epoch])
    """
    num = 2 #移動平均の個数
    b = np.ones(num)/num

    loss_list2 = np.convolve(loss_list, b, mode='same') #移動平均
    val_loss_list2 = np.convolve(val_loss_list, b, mode='same') #移動平均
    #plt.plot(itr_list, loss_list2, label='main/loss', lw=2)
    #plt.plot(itr_list, val_loss_list2, label='validation/main/loss', lw=2)
    """
    # 折れ線グラフを出力
    plt.plot(epoch_list, train_loss_list, '-x', label='train/loss', lw=2)
    plt.plot(epoch_list, val1_loss_list, '-x', label='val1/loss', lw=2)
    plt.plot(epoch_list, val2_loss_list, '-x', label='val2/loss', lw=2)
    # if args.loss_minimam==True:
    #     plt.plot(epoch_list[val1_optimal_epoch], val1_loss_min_value, 'o', markersize=15,color="None", markeredgewidth=3,markeredgecolor="cyan")
    #     plt.plot(epoch_list[val2_optimal_epoch], val2_loss_min_value, 'o', markersize=15,color="None", markeredgewidth=3,markeredgecolor="cyan")
    plt.yscale("linear")
    plt.grid(which="both")
    plt.ylim([0., 0.6])
    # plt.legend()
    # plt.title("loss curve")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(args.input + loss_curve_name)
    plt.close()

    # ACCグラフを出力
    plt.plot(epoch_list, train_acc_list, '-x', label='train/acc', lw=2, color='blue')
    plt.plot(epoch_list, val1_acc_list, '-x', label='val1/acc', lw=2, color='green')
    plt.plot(epoch_list, val2_acc_list, '-x', label='val2/acc', lw=2, color='red')
    # if args.loss_minimam==False:
    #     plt.plot(epoch_list[val1_optimal_epoch], val1_acc_max_value, 'o', markersize=15,color="None", markeredgewidth=3,markeredgecolor="cyan")
    #     plt.plot(epoch_list[val2_optimal_epoch], val2_acc_max_value, 'o', markersize=15,color="None", markeredgewidth=3,markeredgecolor="cyan")

    # plt.plot(epoch_list[np.where(val_acc_list == np.amax(val_acc_list))], val_acc_list[np.argmax()np.amax(val_acc_list)], 'o', markersize=10,color="pink")

    # plt.text(epoch_list[np.where(val_acc_list == np.amax(val_acc_list))], np.amax(val_acc_list),
    #          str(int(epoch_list[np.where(val_acc_list == np.amax(val_acc_list))])), ha='center', va='top',
    #          fontsize=10)

    plt.yscale("linear")
    plt.grid(which="both")
    plt.ylim([0.3, 1.])
    # plt.legend()
    # plt.title("accuracy curve")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.tight_layout()
    plt.savefig(args.input + acc_curve_name)
    plt.close()

    model1=args.input+'gcnn_epoch_{}.npz'.format(int(epoch_list[val1_optimal_epoch]))
    model2=args.input+'gcnn_epoch_{}.npz'.format(int(epoch_list[val2_optimal_epoch]))
    models=[model1,model2]
    test_lists=[test1_list,test2_list]
    out_path_list=[out1_dir,out2_dir]
    max_values=[val1_acc_max_value,val2_acc_max_value]

####################################################################################

    start=time.time()
    A=np.load(data_dir+graph_path)

    for j in range(2):
        out_path=out_path_list[j]
        test_list=test_lists[j]
        model=models[j]
        max_value=max_values[j]

        test = BrainSpectDataset(data_dir=data_dir, data_list_txt=test_list,frip=False,A=A)
        gcnn = GraphCNN(A=A,graph_dir=os.path.join(args.base,args.input),out_dir=None)

        chainer.serializers.load_npz(model, gcnn)
        if args.gpu >= 0:
            chainer.backends.cuda.set_max_workspace_size(2 * 512 * 1024 * 1024)
            chainer.global_config.autotune = True
            chainer.backends.cuda.get_device_from_id(args.gpu).use()
            gcnn.to_gpu()
        xp = gcnn.xp
        path_pairs = []
        prob_list = []
        gt_list = []
        prediction_list = []
        with open(os.path.join(data_dir, test_list)) as paths_file:
            for line in paths_file:
                line = line.split()
                if not line: continue
                path_pairs.append(line[:])

        # print("len(path_pairs)",len(path_pairs))
        # print('path_pairs',path_pairs)
        # path_pairs=np.array(path_pairs)
        # print('path_pairs',path_pairs,path_pairs.shape)


        N_N = 0
        N_A = 0
        N_D = 0
        N_F = 0
        A_N = 0
        A_A = 0
        A_D = 0
        A_F = 0
        D_N = 0
        D_A = 0
        D_D = 0
        D_F = 0
        F_N = 0
        F_A = 0
        F_D = 0
        F_F = 0
        case_N_N = []
        case_N_A = []
        case_N_D = []
        case_N_F = []
        case_A_N = []
        case_A_A = []
        case_A_D = []
        case_A_F = []
        case_D_N = []
        case_D_A = []
        case_D_D = []
        case_D_F = []
        case_F_N = []
        case_F_A = []
        case_F_D = []
        case_F_F = []

        for i, (label,intensity) in enumerate(test):
            # print('Case: {}'.format(test.path_pairs[i][0]))
            # print('intensity',intensity.shape,intensity)
            xp_intensity = chainer.Variable(xp.array(intensity[np.newaxis,:], dtype=xp.float32))
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                probability = gcnn(xp_intensity)
            probability = probability.data

            probability = chainer.cuda.to_cpu(probability)
            probability = np.reshape(probability, [4])

            ground_truth = label
            ground_truth = np.argmax(ground_truth.astype(np.int32))
            prediction = np.argmax(probability, axis=0)

            case = re.findall('[A-Z]-[0-9]+', path_pairs[i][0])
            case=str(case)
            case=case.replace('\'','').replace('[','').replace(']','-x')
            # print(type(case))
            # print(type(case))

            case = case.strip()

            if ground_truth == 0:
                if prediction == 0:
                    N_N += 1
                    case_N_N.append(case)

                if prediction == 1:
                    N_A += 1
                    case_N_A.append(case)

                if prediction == 2:
                    N_D += 1
                    case_N_D.append(case)

                if prediction == 3:
                    N_F += 1
                    case_N_F.append(case)

            if ground_truth == 1:
                if prediction == 0:
                    A_N += 1
                    case_A_N.append(case)
                if prediction == 1:
                    A_A += 1
                    case_A_A.append(case)

                if prediction == 2:
                    A_D += 1
                    case_A_D.append(case)

                if prediction == 3:
                    A_F += 1
                    case_A_F.append(case)

            if ground_truth == 2:
                if prediction == 0:
                    D_N += 1
                    case_D_N.append(case)

                if prediction == 1:
                    D_A += 1
                    case_D_A.append(case)

                if prediction == 2:
                    D_D += 1
                    case_D_D.append(case)

                if prediction == 3:
                    D_F += 1
                    case_D_F.append(case)

            if ground_truth == 3:
                if prediction == 0:
                    F_N += 1
                    case_F_N.append(case)

                if prediction == 1:
                    F_A += 1
                    case_F_A.append(case)

                if prediction == 2:
                    F_D += 1
                    case_F_D.append(case)

                if prediction == 3:
                    F_F += 1
                    case_F_F.append(case)

            prob_list.append((probability))
            gt_list.append((ground_truth))
            prediction_list.append((prediction))

        cases_dict=dict(N_N=case_N_N,N_A=case_N_A,N_D=case_N_D,N_F=case_N_F,
                  A_N=case_A_N,A_A=case_A_A,A_D=case_A_D,A_F=case_A_F,
                  D_N=case_D_N,D_A=case_D_A,D_D=case_D_D,D_F=case_D_F,
                  F_N=case_F_N,F_A=case_F_A,F_D=case_F_D,F_F=case_F_F)
        df=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in cases_dict.items() ]))
        # df = pd.DataFrame([case_N_N, case_N_A, case_N_D, case_N_F,
        #                    case_A_N, case_A_A, case_A_D, case_A_F,
        #                    case_D_N, case_D_A, case_D_D, case_D_F,
        #                    case_F_N, case_F_A, case_F_D, case_F_F]
        #                   , index=['N_N', 'N_A', 'N_D', 'N_F',
        #                            'A_N', 'A_A', 'A_D', 'A_F',
        #                            'D_N', 'D_A', 'D_D', 'D_F',
        #                            'F_N', 'F_A', 'F_D', 'F_F', ])
        df.to_csv(os.path.join(args.base, out_path) + '/cases.csv')

        prob_list = np.array(prob_list)
        gt_list = np.array(gt_list)
        prediction_list = np.array(prediction_list)

        acc = F.accuracy(prob_list, gt_list)
        f = open(os.path.join(args.base, out_path) + filename, 'w', newline='')
        writer = csv.writer(f,
                            # lineterminator='\n'
                            )
        # f.write('probability\n')
        f.write(models[j].replace(args.input,'').replace('.npz',''))
        f.write(' was selected')
        f.write('\naccuracy {}fold_group{}_test{}\n'.format(fold,group,j+1))
        f.write('validation_max_accuracy={}\n'.format(max_value))
        f.write('test_accuracy=')
        writer.writerow([acc])  # [ground_truth], [probability]
        f.write('\n\n')

        prediction_one_hot = np.zeros((len(prob_list), num_of_label))
        gt_one_hot = np.zeros((len(prob_list), num_of_label))

        for n in range(len(prob_list)):
            test._path_pairs
            case = test._path_pairs[n][5].replace('data/preprocessed/', '')
            case = case.replace('.npy', '')
            f.write(case)
            f.write("\n")
            writer.writerow([prob_list[n][0], prob_list[n][1], prob_list[n][2], prob_list[n][3]])
            f.write('prediction')
            f.write("\n")
            prediction_one_hot[n][prediction_list[n]] = 1
            gt_one_hot[n][gt_list[n]] = 1
            writer.writerow([prediction_one_hot[n][0], prediction_one_hot[n][1], prediction_one_hot[n][2], prediction_one_hot[n][3]])
            f.write('groud_truth')
            f.write("\n")
            writer.writerow([gt_one_hot[n][0], gt_one_hot[n][1], gt_one_hot[n][2], gt_one_hot[n][3]])
            f.write("\n")

        f.close()

        # print(N_N, N_A, N_D, N_F)
        f = open(os.path.join(args.base, out_path) +matrix_filename, 'w', newline='')
        f.write('{}fold_group{}_test{}\n'.format(fold,group,j+1))
        f.close()

        df = pd.DataFrame(data=[[N_N, N_A, N_D, N_F], [A_N, A_A, A_D, A_F], [D_N, D_A, D_D, D_F], [F_N, F_A, F_D, F_F]],
                          index=['HC', 'AD', 'DLB', 'FTD'], columns=['HC', 'AD', 'DLB', 'FTD']
                          )
        df.to_csv(os.path.join(args.base, out_path) + matrix_filename)

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    time_filename="elapsed_time.csv"
    f=open(os.path.join(args.base, out_path) +time_filename, 'w', newline='')
    f.write("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    f.close()

if __name__ == '__main__':
    main()
