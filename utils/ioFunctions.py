# coding:utf-8
"""
@auther tzw
"""
import os, sys, time
import numpy as np
import pandas as pd
import yaml
import pickle
import re

def read_dat(path):
    """
    @path: file path
    @return: pandas dataframe
    """
    root, ext = os.path.splitext(path)
    if not ext == '.dat':
        raise NotImplementedError()

    df = pd.read_csv(path, names=('x', 'y', 'z','intensity'))

    return df

def save_args(output_dir, args):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open('{}/config.yml'.format(output_dir), 'w') as f:
        f.write(yaml.dump(vars(args), default_flow_style=False))

def read_data_list(path):
    root, ext = os.path.splitext(path)
    if not ext == '.txt':
        raise NotImplementedError()

    data_list = []
    with open(path) as paths_file:
        for line in paths_file:
            # line = line.split()
            # if not line : continue
            if not line: continue
            line = line.replace('\n','')
            data_list.append(line[:])

    return data_list

def read_pickle_data(input_dir, group_num):
    with open('{}/grouped_graph_{}'.format(input_dir, group_num), 'rb') as handle:
        batch_laplacian = pickle.load(handle)

    with open('{}/grouped_intensity_{}'.format(input_dir, group_num), 'rb') as handle:
        batch_intensity = pickle.load(handle)

    with open('{}/grouped_label_{}'.format(input_dir, group_num), 'rb') as handle:
        batch_label = pickle.load(handle)

    return batch_laplacian, batch_intensity, batch_label

def load_training_data(root,txt_path):
    root = root
    label_dtype = np.int32
    dtype = np.float32

    path_pairs = []
    with open(txt_path) as paths_file:
        for line in paths_file:
            line = line.split()  # [][2]の配列にしてる？
            if not line: continue
            path_pairs.append(line[:])

    num_of_case = len(path_pairs)
    print('    # of cases: {}'.format(num_of_case))

    dataset = []
    intensity = []
    intensity_training=[]

    intensity_N=[]
    intensity_A=[]
    intensity_D=[]
    intensity_F=[]

    for i in path_pairs:
        print('   Org   from: {}'.format(i[0]))
        print('   label from: {}'.format(i[5]))
        # Read data
        intensity0 = pd.read_csv(os.path.join(root, i[0]))['intensity'].values.astype(dtype)
        intensity1 = pd.read_csv(os.path.join(root, i[1]))['intensity'].values.astype(dtype)
        intensity2 = pd.read_csv(os.path.join(root, i[2]))['intensity'].values.astype(dtype)
        intensity3 = pd.read_csv(os.path.join(root, i[3]))['intensity'].values.astype(dtype)
        intensity4 = pd.read_csv(os.path.join(root, i[4]))['intensity'].values.astype(dtype)
        intensity = np.array([intensity0, intensity1, intensity2, intensity3, intensity4])
        print('intensity.shape', intensity.shape)
        # org = org[np.newaxis, :]#(ch, z, y, x)

        label = np.load(os.path.join(root, i[5])).astype(label_dtype)
        dataset.append((intensity, label))

        # intensity_training.append(intensity)
        if label[0]==1:
            intensity_N.append(intensity)
        if label[1]==1:
            intensity_A.append(intensity)
        if label[2] ==1:
            intensity_D.append(intensity)
        if label[3] ==1:
            intensity_F.append(intensity)

    intensity_N = np.array(intensity_N)
    intensity_A = np.array(intensity_A)
    intensity_D = np.array(intensity_D)
    intensity_F = np.array(intensity_F)

    # num_of_cases, num_of_input_channel, num_of_nodes = intensity_training.shape
    # intensity_training = intensity_training.transpose(2, 0, 1)

    return intensity_N, intensity_A, intensity_D, intensity_F

def load_training_data_and_case(root, txt_path):
    root = root
    label_dtype = np.int32
    dtype = np.float32

    path_pairs = []
    with open(txt_path) as paths_file:
        for line in paths_file:
            line = line.split()  # [][2]の配列にしてる？
            if not line: continue
            path_pairs.append(line[:])

    num_of_case = len(path_pairs)
    print('    num of cases: {}'.format(num_of_case))

    dataset = []
    intensity = []
    intensity_training = []

    intensity_N = []
    intensity_A = []
    intensity_D = []
    intensity_F = []

    case_N = []
    case_A = []
    case_D = []
    case_F = []

    for i in path_pairs:
        print('   Org   from: {}'.format(i[0]))
        print('   label from: {}'.format(i[5]))
        # Read data
        intensity0 = pd.read_csv(os.path.join(root, i[0]))['intensity'].values.astype(dtype)
        intensity1 = pd.read_csv(os.path.join(root, i[1]))['intensity'].values.astype(dtype)
        intensity2 = pd.read_csv(os.path.join(root, i[2]))['intensity'].values.astype(dtype)
        intensity3 = pd.read_csv(os.path.join(root, i[3]))['intensity'].values.astype(dtype)
        intensity4 = pd.read_csv(os.path.join(root, i[4]))['intensity'].values.astype(dtype)
        intensity = np.array([intensity0, intensity1, intensity2, intensity3, intensity4])
        print('intensity.shape', intensity.shape)
        # org = org[np.newaxis, :]#(ch, z, y, x)

        case = re.findall('[A-Z]-[0-9]+', i[5])
        print("case", case)

        label = np.load(os.path.join(root, i[5])).astype(label_dtype)
        dataset.append((intensity, label))

        # intensity_training.append(intensity)
        if label[0] == 1:
            intensity_N.append(intensity)
            case_N.append(case)
        if label[1] == 1:
            intensity_A.append(intensity)
            case_A.append(case)
        if label[2] == 1:
            intensity_D.append(intensity)
            case_D.append(case)
        if label[3] == 1:
            intensity_F.append(intensity)
            case_F.append(case)

    intensity_N = np.array(intensity_N)
    intensity_A = np.array(intensity_A)
    intensity_D = np.array(intensity_D)
    intensity_F = np.array(intensity_F)

    case_N = np.array(case_N)
    num_N = case_N.shape[0]
    case_N = case_N.reshape(num_N)

    case_A = np.array(case_A)
    num_A = case_A.shape[0]
    case_A = case_A.reshape(num_A)

    case_D = np.array(case_D)
    num_D = case_D.shape[0]
    case_D = case_D.reshape(num_D)

    case_F = np.array(case_F)
    num_F = case_F.shape[0]
    case_F = case_F.reshape(num_F)
    # num_of_cases, num_of_input_channel, num_of_nodes = intensity_training.shape
    # intensity_training = intensity_training.transpose(2, 0, 1)

    return intensity_N, intensity_A, intensity_D, intensity_F \
        , case_N, case_A, case_D, case_F


def load_intensity_and_label(root, txt_path):
    root = root
    label_dtype = np.int32
    dtype = np.float32

    path_pairs = []
    with open(txt_path) as paths_file:
        for line in paths_file:
            line = line.split()  # [][2]の配列にしてる？
            if not line: continue
            path_pairs.append(line[:])

    num_of_case = len(path_pairs)
    print('    num of cases: {}'.format(num_of_case))

    dataset = []
    intensity = []
    intensity_training = []

    intensities = []
    labels = []

    for i in path_pairs:
        print('   Org   from: {}'.format(i[0]))
        print('   label from: {}'.format(i[5]))
        # Read data
        intensity0 = pd.read_csv(os.path.join(root, i[0]))['intensity'].values.astype(dtype)
        intensity1 = pd.read_csv(os.path.join(root, i[1]))['intensity'].values.astype(dtype)
        intensity2 = pd.read_csv(os.path.join(root, i[2]))['intensity'].values.astype(dtype)
        intensity3 = pd.read_csv(os.path.join(root, i[3]))['intensity'].values.astype(dtype)
        intensity4 = pd.read_csv(os.path.join(root, i[4]))['intensity'].values.astype(dtype)
        intensity = np.array([intensity0, intensity1, intensity2, intensity3, intensity4])
        print('intensity.shape', intensity.shape)
        label = np.load(os.path.join(root, i[5])).astype(label_dtype)
        intensities.append([intensity])

        labels.append([label])


    intensities=np.array(intensities)
    num_cases, hoge, in_channel, signal_dimension = intensities.shape
    intensities = np.reshape(intensities, (num_cases, in_channel, signal_dimension))

    labels = np.array(labels)
    labels = np.reshape(labels, (num_cases, 4))
    labels=np.argmax(labels,axis=1)
    # for i in range(num_cases):
    #     labels[i]=np.argmax(labels[i])

    return intensities,labels

def load_training_data_and_case(root, txt_path):
    root = root
    label_dtype = np.int32
    dtype = np.float32

    path_pairs = []
    with open(txt_path) as paths_file:
        for line in paths_file:
            line = line.split()  # [][2]の配列にしてる？
            if not line: continue
            path_pairs.append(line[:])

    num_of_case = len(path_pairs)
    print('    num of cases: {}'.format(num_of_case))

    dataset = []
    intensity = []
    intensity_training = []

    intensity_N = []
    intensity_A = []
    intensity_D = []
    intensity_F = []

    case_N = []
    case_A = []
    case_D = []
    case_F = []

    for i in path_pairs:
        print('   Org   from: {}'.format(i[0]))
        print('   label from: {}'.format(i[5]))
        # Read data
        intensity0 = pd.read_csv(os.path.join(root, i[0]))['intensity'].values.astype(dtype)
        intensity1 = pd.read_csv(os.path.join(root, i[1]))['intensity'].values.astype(dtype)
        intensity2 = pd.read_csv(os.path.join(root, i[2]))['intensity'].values.astype(dtype)
        intensity3 = pd.read_csv(os.path.join(root, i[3]))['intensity'].values.astype(dtype)
        intensity4 = pd.read_csv(os.path.join(root, i[4]))['intensity'].values.astype(dtype)
        intensity = np.array([intensity0, intensity1, intensity2, intensity3, intensity4])
        print('intensity.shape', intensity.shape)
        # org = org[np.newaxis, :]#(ch, z, y, x)

        case = re.findall('[A-Z]-[0-9]+', i[5])
        print("case", case)

        label = np.load(os.path.join(root, i[5])).astype(label_dtype)
        dataset.append((intensity, label))

        # intensity_training.append(intensity)
        if label[0] == 1:
            intensity_N.append(intensity)
            case_N.append(case)
        if label[1] == 1:
            intensity_A.append(intensity)
            case_A.append(case)
        if label[2] == 1:
            intensity_D.append(intensity)
            case_D.append(case)
        if label[3] == 1:
            intensity_F.append(intensity)
            case_F.append(case)

    intensity_N = np.array(intensity_N)
    intensity_A = np.array(intensity_A)
    intensity_D = np.array(intensity_D)
    intensity_F = np.array(intensity_F)

    case_N = np.array(case_N)
    num_N = case_N.shape[0]
    case_N = case_N.reshape(num_N)

    case_A = np.array(case_A)
    num_A = case_A.shape[0]
    case_A = case_A.reshape(num_A)

    case_D = np.array(case_D)
    num_D = case_D.shape[0]
    case_D = case_D.reshape(num_D)

    case_F = np.array(case_F)
    num_F = case_F.shape[0]
    case_F = case_F.reshape(num_F)
    # num_of_cases, num_of_input_channel, num_of_nodes = intensity_training.shape
    # intensity_training = intensity_training.transpose(2, 0, 1)

    return intensity_N, intensity_A, intensity_D, intensity_F \
        , case_N, case_A, case_D, case_F


def load_dataset_and_case(root, txt_path):
    root = root
    label_dtype = np.int32
    dtype = np.float32


    path_pairs = []
    with open(os.path.join(root,txt_path)) as paths_file:
        for line in paths_file:
            line = line.split()  # [][2]の配列にしてる？
            if not line: continue
            path_pairs.append(line[:])

    num_of_case = len(path_pairs)
    print('    num of cases: {}'.format(num_of_case))

    dataset = []
    intensity = []
    intensity_training = []
    cases=[]

    intensities = []
    labels = []

    for i in path_pairs:
        print('   Org   from: {}'.format(i[0]))
        print('   label from: {}'.format(i[5]))
        # Read data
        intensity0 = pd.read_csv(os.path.join(root, i[0]))['intensity'].values.astype(dtype)
        intensity1 = pd.read_csv(os.path.join(root, i[1]))['intensity'].values.astype(dtype)
        intensity2 = pd.read_csv(os.path.join(root, i[2]))['intensity'].values.astype(dtype)
        intensity3 = pd.read_csv(os.path.join(root, i[3]))['intensity'].values.astype(dtype)
        intensity4 = pd.read_csv(os.path.join(root, i[4]))['intensity'].values.astype(dtype)
        intensity = np.array([intensity0, intensity1, intensity2, intensity3, intensity4])
        print('intensity.shape', intensity.shape)
        label = np.load(os.path.join(root, i[5])).astype(label_dtype)
        intensities.append([intensity])

        labels.append([label])

        case = re.findall('[A-Z]-[0-9]+', i[5])
        print("case", case)
        cases.append(case)


    intensities=np.array(intensities)
    num_cases, hoge, in_channel, signal_dimension = intensities.shape
    intensities = np.reshape(intensities, (num_cases, in_channel, signal_dimension))

    labels = np.array(labels)
    labels = np.reshape(labels, (num_cases, 4))
    labels=np.argmax(labels,axis=1)

    cases = np.array(cases)
    num_cases = cases.shape[0]
    cases = cases.reshape(num_cases)


    return intensities,labels,cases