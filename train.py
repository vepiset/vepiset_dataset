import cv2
import numpy as np
import pandas as pd

from base_trainer.net_work import Train
from base_trainer.dataietr import AlaskaDataIter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from train_config import config as cfg



def main():
    n_fold = 5

    def get_fold(n_fold=n_fold):

        data = pd.read_csv(cfg.DATA.data_file)

        folds = data.copy()
        Fold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=cfg.SEED)
        for n, (train_index, val_index) in enumerate(Fold.split(folds, folds['target'])):
            folds.loc[val_index, 'fold'] = int(n)
        return folds

    data = get_fold(n_fold)

    for fold in range(n_fold):
        ###build dataset

        train_ind = data[data['train_val'] == 0].index.values
        train_data = data.iloc[train_ind].copy()
        val_ind = data[data['train_val'] == 1].index.values
        val_data = data.iloc[val_ind].copy()

        ###build trainer

        if cfg.TRAIN.vis:
            print('show it, here')

            trainds = AlaskaDataIter(train_data, training_flag=True, shuffle=False)
            train_ds = DataLoader(trainds,
                                  cfg.TRAIN.batch_size,
                                  num_workers=cfg.TRAIN.process_num,
                                  shuffle=True)

            for images, labels in train_ds:

                for i in range(images.shape[0]):
                    example_image = np.array(images[i], dtype=np.uint8)
                    example_image = np.transpose(example_image, [1, 2, 0])
                    example_label = np.array(labels[i])

                    print(example_label)
                    print(example_label.shape)
                    cv2.imshow('example', example_image)

                    cv2.waitKey(0)


        trainer = Train(train_df=train_data,
                        val_df=val_data,
                        fold=fold)

        ### train
        trainer.custom_loop()



if __name__ == '__main__':
    main()