# coding:utf-8
import os, sys, time
import numpy as np
import chainer
import scipy
import pandas as pd
import sklearn.metrics.pairwise
from scipy.sparse import csr_matrix

def distance_sklearn_metrics(z, k=4, metric='euclidean'):
    """Compute exact pairwise distances."""
    # Adapted from https://github.com/mdeff/cnn_graph/blob/master/lib/graph.py
    d = sklearn.metrics.pairwise.pairwise_distances(
        z, metric=metric, n_jobs=-2)
    # k-NN graph.
    idx = np.argsort(d)[:, 1:k + 1]
    d.sort()
    d = d[:, 1:k + 1]
    return d, idx

class BrainSpectDataset(chainer.dataset.DatasetMixin):
    def __init__(self, data_dir, data_list_txt, frip, A):
        """
        """
        self._root = data_dir
        self._label_dtype = np.int32
        self._dtype = np.float32
        self._frip=frip
        self._A=A
        data_list_txt=data_dir+data_list_txt

        if self._frip==True:
            self._frip_ind=np.load(self._root+"/frip_ind.npy")

        path_pairs = []
        with open(data_list_txt) as paths_file:
            for line in paths_file:
                line = line.split()  # [][2]の配列にしてる？
                if not line: continue
                path_pairs.append(line[:])

        self._num_of_case = len(path_pairs)
        print('    # of cases: {}'.format(self._num_of_case))

        self._dataset = []
        intensities=[]
        labels=[]
        # for i in path_pairs[0:10]:

        for i in path_pairs:
            print('   Org   from: {}'.format(i[0]))
            print('   label from: {}'.format(i[5]))
            # Read data
            intensity0 =pd.read_csv(os.path.join(self._root, i[0]))['intensity'].values.astype(self._dtype)
            intensity1 = pd.read_csv(os.path.join(self._root, i[1]))['intensity'].values.astype(self._dtype)
            intensity2 = pd.read_csv(os.path.join(self._root, i[2]))['intensity'].values.astype(self._dtype)
            intensity3 = pd.read_csv(os.path.join(self._root, i[3]))['intensity'].values.astype(self._dtype)
            intensity4 = pd.read_csv(os.path.join(self._root, i[4]))['intensity'].values.astype(self._dtype)
            intensity = np.array([intensity0, intensity1, intensity2, intensity3, intensity4])
            print('intensity.shape', intensity.shape)
            label = np.load(os.path.join(self._root, i[5])).astype(self._label_dtype)
            # self._dataset.append([intensity, label])
            intensities.append([intensity])
            labels.append([label])


        #######################

        # A=np.load("D:/PycharmProjects/GraphCNN-for-Brain-Spect/results/generated_graph/sotsuron_k8_J_10000_20200103_soturonmax.npy")
        # A=np.load("D:/PycharmProjects/GraphCNN-for-Brain-Spect/results/generated_graph/sotsuron_OR_1000_20191020_conservation_oldmax.npy")
        # A=mmread("D:/PycharmProjects/GraphCNN-for-Brain-Spect/results/generated_graph/graph_sotsuron_separated_isolated.mtx")
        # A=A+A.T

        row0 = np.all(A == 0, axis=0)
        ind0 = np.where(row0 == True)
        row_pos = np.any(A > 0, axis=0)
        row_ind_pos = np.where(row_pos == True)
        col_pos=np.any(A>0,axis=1)
        col_ind_pos= np.where(col_pos == True)
        A=np.delete(A,ind0,axis=0)
        A=np.delete(A,ind0,axis=1)
        # A=A[ind_pos][:]
        intensities=np.array(intensities)
        intensities=np.delete(intensities,ind0,axis=3)
        num_cases,hoge,in_channel,signal_dimension=intensities.shape
        intensities=np.reshape(intensities,(num_cases,in_channel,signal_dimension))
        labels=np.array(labels)
        labels=np.reshape(labels,(num_cases,4))
        for i in range(len(intensities)):
            self._dataset.append((intensities[i],labels[i]))
        # df=pd.DataFrame(A)
        # df.to_csv("adjacency_matrix_downsized.csv")
        # df2=pd.DataFrame(intensities[0])
        # df2.to_csv("_input_signal_downsized.csv")


        # print("intensities",intensities[0][0].shape)
        # print("self._dataset_intensity",self._dataset[0][0].shape)

        # np.delete(self._dataset[:][0],ind0,axis=0)
        # print("type",type(self._dataset),type(self._dataset[0][0]))

        self.A=scipy.sparse.csr_matrix(A)
        self._path_pairs=path_pairs


    def __len__(self):
        return len(self._dataset)

    def get_example(self, i):
        intensity,label=self._dataset[i]

        if self._frip==True:
            if np.random.rand()>0.5:
                intensity=intensity[:,self._frip_ind]

        return label,intensity


# class BrainSpectDataset_predict(chainer.dataset.DatasetMixin):
#     def __init__(self, root, data_list_txt):
#         """
#         """
#         self._root = root
#         self._label_dtype = np.int32
#         self._dtype = np.float32
#
#         path_pairs = []
#         with open(data_list_txt) as paths_file:
#             for line in paths_file:
#                 line = line.split()  # [][2]の配列にしてる？
#                 if not line: continue
#                 path_pairs.append(line[:])
#
#         self._num_of_case = len(path_pairs)
#         print('    # of cases: {}'.format(self._num_of_case))
#
#         self._dataset = []
#         for i in path_pairs:
#             print('   Org   from: {}'.format(i[0]))
#             print('   label from: {}'.format(i[5]))
#             # Read data
#             intensity0 = pd.read_csv(os.path.join(self._root, i[0]))['intensity'].values.astype(self._dtype)
#             intensity1 = pd.read_csv(os.path.join(self._root, i[1]))['intensity'].values.astype(self._dtype)
#             intensity2 = pd.read_csv(os.path.join(self._root, i[2]))['intensity'].values.astype(self._dtype)
#             intensity3 = pd.read_csv(os.path.join(self._root, i[3]))['intensity'].values.astype(self._dtype)
#             intensity4 = pd.read_csv(os.path.join(self._root, i[4]))['intensity'].values.astype(self._dtype)
#             intensity = np.array([intensity0, intensity1, intensity2, intensity3, intensity4])
#             print('intensity.shape', intensity.shape)
#             # org = org[np.newaxis, :]#(ch, z, y, x)
#
#             label = np.load(os.path.join(self._root, i[5])).astype(self._label_dtype)
#             self._dataset.append((intensity, label))
#
#         self._path_pairs=path_pairs
#
#     def __len__(self):
#         return len(self._dataset)
#
#     def get_example(self, i):
#         return self._dataset[i][1],self._dataset[i][0]
#
