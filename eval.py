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


def get_model(weight, device, is_base):
    channel_num = 0
    if is_base == 0:
        channel_num = 128
    model = Net(add_channel=channel_num).to(device)
    state_dict = torch.load(weight, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model



def eval_add_plt(weight_video, weight_base, test_path):
    rocauc_score = ROCAUCMeter()

    base_y_true, base_y_pre = estimated_score(weight_base, test_path, 1)
    video_y_true, video_y_pre = estimated_score(weight_video, test_path, 0)

    print("========= estimated_score base line  ==========", test_path)
    rocauc_score.report_with_recall_precision(base_y_true, base_y_pre)
    #rocauc_score.report_all(base_y_true, base_y_pre)

    print("========= estimated_score add video ==========", test_path)
    rocauc_score.report_with_recall_precision(video_y_true, video_y_pre)
    #rocauc_score.report_all(video_y_true, video_y_pre)



    print("========= precision_recall ==========", test_path)
    img_path_p_r = test_path.split(".")[0] + "_Precision_Recall__Add_Data_Pre" + ".jpg"
    rocauc_score.report_with_recall(video_y_true, video_y_pre, base_y_true, base_y_pre, img_path_p_r)

    print("========= Specificity_Sensitivity ==========", test_path)
    img_path_t_f = test_path.split(".")[0] + "_Specificity_Sensitivity__Add_Data_Pre" + ".jpg"
    rocauc_score.report_tpr_fpr(video_y_true, video_y_pre, base_y_true, base_y_pre, img_path_t_f)



def estimated_score(weight, test_path, is_base):
   # print("========= estimated_score test_path ==========", test_path)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    rocauc_score = ROCAUCMeter()
    model = get_model(weight, device, is_base)
    val_ds = get_data_iter(test_path)

    labels_list = []
    y_pre_list = []

    y_true_11 = None
    y_pred_11 = None

    with torch.no_grad():
        print("val_ds:", val_ds)
        for (images, labels, video_feature) in tqdm(val_ds):
            data = images.to(device).float()
            labels = labels.to(device).float()
            labels_list.append(labels)
            # base_feature = base_feature.to(device).float()
            video_feature = video_feature.to(device).float()
            batch_size = data.shape[0]
            predictions = model(data, video_feature, is_base)
            y_pre_list.append(predictions)
            y_true_11, y_pred_11 = rocauc_score.update(labels, predictions)
            #print("=====y_true_11=====", y_true_11)
            #print("=====y_pred_11=====", y_pred_11)
    # save labels_list and y_pre_list
    labels_data = torch.cat(labels_list, dim=0)
    y_pre_data = torch.cat(y_pre_list, dim=0)
    print("labels len:", len(labels_data.tolist()))
    print("predictions len:", len(y_pre_data.tolist()))

    return y_true_11, y_pred_11



def eval(weight, test_path):

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    rocauc_score = ROCAUCMeter()
    model = get_model(weight, device)
    val_ds = get_data_iter(test_path)

    labels_list = []
    y_pre_list = []

    with torch.no_grad():
        print("val_ds:", val_ds)
        for (images, labels) in tqdm(val_ds):

            data = images.to(device).float()
            labels = labels.to(device).float()

            labels_list.append(labels)
            batch_size = data.shape[0]
            predictions = model(data)
            intermediate_output = model.intermediate_layer(data)
            y_pre_list.append(predictions)
            rocauc_score.update(labels, predictions)

        labels_data = torch.cat(labels_list, dim=0)
        y_pre_data = torch.cat(y_pre_list, dim=0)
        rocauc_score.report_with_recall_precision()

    print("labels len:", len(labels_data.tolist()))
    print("predictions len:", len(y_pre_data.tolist()))


    return rocauc_score


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Start train.')

    parser.add_argument('--weight', dest='weight', type=str, default=None, \
                        help='the weight to use')
    parser.add_argument('--test_path', dest='test_path', type=str, default=None, \
                        help='the weight to use')

    args = parser.parse_args()
    weight = args.weight
    test_path = args.test_path
    try:
        eval(weight, test_path)
    except Exception as e:
        print("=====e=====", e)
