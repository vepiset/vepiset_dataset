import random
import copy
import numpy as np
import os
from utils.logger import logger



class AlaskaDataIter():
    def __init__(self, df,
                 training_flag=True, shuffle=True):

        self.training_flag = training_flag
        self.shuffle = shuffle
        self.raw_data_set_size = None


        self.df = df
        logger.info(' contains%d samples  %d pos' % (len(self.df), np.sum(self.df['target'] == 1)))
        logger.info(' contains%d samples' % len(self.df))

        logger.info(' After filter contains%d samples  %d pos' % (len(self.df), np.sum(self.df['target'] == 1)))
        logger.info(' After filter contains%d samples' % len(self.df))

        self.leads_nm = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5',
                         'T6',
                         'Fz', 'Cz', 'Pz',
                         'PG1', 'PG2', 'A1', 'A2',
                         'EKG1', 'EKG2', 'EMG1',
                         'EMG2', 'EMG3', 'EMG4']

        self.leads_dict = {value: index for index, value in enumerate(self.leads_nm)}


        self.left_brain = ['Fp1', 'F3', 'C3', 'P3', 'O1', 'T5', 'T3', 'F7']
        self.right_brain = ['Fp2', 'F4', 'C4', 'P4', 'O2', 'T6', 'T4', 'F8']

    def filter(self, df):

        df = copy.deepcopy(df)
        pos_indx = df['target'] == 1
        pos_df = df[pos_indx]

        neg_indx = df['target'] == 0
        neg_df = df[neg_indx]

        neg_df = neg_df.sample(frac=1)

        dst_df = neg_df
        for i in range(1):
            dst_df = dst_df._append(pos_df)
        dst_df.reset_index()

        return dst_df

    def __getitem__(self, item):
        return self.single_map_func(self.df.iloc[item], self.training_flag)

    def __len__(self):
        return len(self.df)

    def norm(self, wave):

        wave[:23, ...] = wave[:23, ...] / 1e-3
        wave[23:, ...] = wave[23:, ...] / 1e-2

        # 心电和肌电
        heart_wave = wave[23, :] - wave[24, :]

        muscle_wave1 = wave[25, :] - wave[26, :]

        muscle_wave2 = wave[27, :] - wave[28, :]

        heart_muscle = np.stack([heart_wave, muscle_wave1, muscle_wave2], axis=0)

        wave_26 = np.concatenate([wave[:23, ...], heart_muscle], axis=0)

        return wave_26

    def roll(self, waves, strength=2000 // 2):

        start = random.randint(-strength, strength)
        waves = np.roll(waves, start, axis=1)

        return waves

    def xshuffle(self, wave):

        n_channels, n_samples = wave.shape

        channel_indices = np.arange(n_channels)

        np.random.shuffle(channel_indices)

        shuffled_wave = wave[channel_indices]

        return shuffled_wave

    def avg_lead(self, waves):

        # copy一份，防止原地修改
        waves = copy.deepcopy(waves)

        meadn = np.mean(waves[:19, :], axis=0)
        data = waves[:19, :] - meadn
        return data


    def lead(self, waves):

        avg_lead = self.avg_lead(waves)

        return avg_lead


    def single_map_func(self, dp, is_training):
        """Data augmentation function."""
        fname = dp['file_path']
        label = dp['target']
        try:
            fname = fname.strip()
            waves = np.load(fname)


        except Exception as e:
            print("=====fname====exception:", fname, e)
            waves = np.zeros(shape=[29, 2000])
            label = 0


        waves = self.norm(waves)

        avg_lead = self.lead(waves)

        if is_training and random.uniform(0, 1) < 1.:
            waves[:19, :] = self.xshuffle(waves[:19, :])
            avg_lead = self.xshuffle(avg_lead)

        waves = np.concatenate([waves, avg_lead], axis=0)


        label = np.expand_dims(label, -1)

        C, L = waves.shape

        if L < 2000:
            waves = np.pad(waves, ((0, 0), (0, 2000 - L)), 'constant', constant_values=0)
        elif L > 2000:
            waves = waves[:, 2000]
        waves = np.ascontiguousarray(waves)

        return waves, label
