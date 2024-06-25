import os
import pandas as pd
import argparse
import numpy as np


def find_npy_files(folder_path):
    npy_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.npy'):
                npy_files.append(os.path.join(root, file))

    return npy_files


def get_data(data_dir):
    samples = find_npy_files(data_dir)
    labels = [1 if int(x.split("__")[1].split(".")[0]) > 1 else int(x.split("__")[1].split(".")[0]) for x in samples]
    train_val = [0] * len(samples)
    return samples, labels, train_val


def get_data_files(numpy_files_path):
    #data_dir = npy_val_data
    fns_list = []
    labels_list = []
    train_val_list = []
    samples, labels, train_val = get_data(numpy_files_path)
    fns_list.extend(samples)
    labels_list.extend(labels)
    train_val_list.extend(train_val)


    return fns_list, labels_list, train_val_list

def main():
    parser = argparse.ArgumentParser(description='start process data')
    parser.add_argument('--numpy_files_path', dest='numpy_files_path', type=str, default=None, \
                        help='the path of numpy files')
    parser.add_argument('--save_csv_file', dest='save_csv_file', type=str, default=None, \
                        help='the path of save csv file')

    args = parser.parse_args()
    numpy_files_path = args.numpy_files_path
    save_csv_file = args.save_csv_file


    val_fns_list, val_labels_list, val_vals_list = get_data_files(numpy_files_path)

    submission = pd.DataFrame({'file_path': val_fns_list,
                               'target': val_labels_list,
                               'train_val': val_vals_list})
    ### split train - val = 8:2
    indices = submission[submission['train_val'] == 0].index
    val_num = len(indices) // 5
    indices_to_change = np.random.choice(indices, val_num, replace=False)
    submission.loc[indices_to_change, 'train_val'] = 1

    print("fns len:", len(val_fns_list))
    print("label len:", len(val_labels_list))
    print("val len:", val_num)
    print("train:val {0}:{1}".format(8, 2))
    submission.to_csv(save_csv_file, index=False)
    submission.head()

if __name__ == '__main__':
    main()



