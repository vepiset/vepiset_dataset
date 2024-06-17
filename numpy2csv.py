import os
import pandas as pd
import argparse



def find_npy_files(folder_path):
    npy_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.npy'):
                npy_files.append(os.path.join(root, file))

    return npy_files


def get_data(data_dir, is_val=0):
    samples = find_npy_files(data_dir)
    labels = [x.split("__")[1].split(".")[0] for x in samples]
    train_val = [is_val] * len(samples)
    return samples, labels, train_val


def get_data_files(numpy_files_path, is_val=0):
    #data_dir = npy_val_data
    fns_list = []
    labels_list = []
    train_val_list = []
    samples, labels, train_val = get_data(numpy_files_path, is_val)
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


    val_fns_list, val_labels_list, val_vals_list = get_data_files(numpy_files_path, is_val=0)

    print("fns len:", len(val_fns_list))
    print("label len:", len(val_labels_list))
    print("val len:", len(val_vals_list))

    submission = pd.DataFrame({'file_path': val_fns_list,
                               'target': val_labels_list,
                               'train_val': val_vals_list})

    submission.to_csv(save_csv_file, index=False)
    submission.head()

if __name__ == '__main__':
    main()



