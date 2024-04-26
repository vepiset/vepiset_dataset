import sys
import torch
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader
from base_trainer.dataietr import AlaskaDataIter
from train_config import config as cfg
from base_trainer.model import Net
from base_trainer.metric import *


sys.path.append('.')


def get_data_iter(test_path=cfg.DATA.data_file):

    data = pd.read_csv(test_path)

    val_ind = data[data['train_val'] == 1].index.values
    val_data = data.iloc[val_ind].copy()

    valds = AlaskaDataIter(val_data, training_flag=False, shuffle=False)
    valds = DataLoader(valds,
                       32,
                       num_workers=2,
                       shuffle=False)
    return valds


def get_model(weight, device):

    model = Net().to(device)
    state_dict = torch.load(weight, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


def eval(weight, test_path):

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    rocauc_score = ROCAUCMeter()
    model = get_model(weight, device)
    val_ds = get_data_iter(test_path)

    labels_list = []
    y_pre_list = []

    with torch.no_grad():
        print("val_ds:", val_ds)
        for (images, labels, video_feature) in tqdm(val_ds):

            data = images.to(device).float()
            labels = labels.to(device).float()

            labels_list.append(labels)
            batch_size = data.shape[0]
            predictions = model(data)
            y_pre_list.append(predictions)
            rocauc_score.update(labels, predictions)

    return rocauc_score


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Start train.')

    parser.add_argument('--weight', dest='weight', type=str, default=None, \
                        help='the weight to use')
    parser.add_argument('--test_path', dest='test_path', type=str, default=None, \
                        help='the weight to use')
    parser.add_argument('--is_base', dest='is_base', type=int, default=0 , \
                        help='the weight to use')

    args = parser.parse_args()
    weight = args.weight
    weight_video = args.weight_video
    weight_base = args.weight_base
    is_base = args.is_base
    test_list = ['test.csv']
    for test_path in test_list:

        try:
            eval(weight, test_path)
        except Exception as e:
            print("=====e=====", e)
