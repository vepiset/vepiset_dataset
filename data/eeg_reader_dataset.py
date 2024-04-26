# -*- coding: utf-8 -*-
import os
import json
import traceback
import mne
import numpy as np

from pathlib import Path
from datetime import datetime, timedelta

from utils.logger import logger
import data.enhanced_nihon as enhanced_nihon

from data.parseLOG import parseSubLOG

label_style_dic = {}


def is_abnormal_label(s: str):
    s = s.strip()

    if 'Waking' in s:
        return True

    if 'Stage' in s:
        return True

    if s in (
            '右中央著尖慢波类周期样出现',
            '左中央、顶、中额为著频繁放电',
            '频繁放电',
            '右前颞频繁可见高波幅慢波、尖波、尖慢复合波',
            '环联针锋相对', '右颞尖波慢波', '左前额、前颞尖慢波'
    ):
        return True
    '''
    if s in (
            'S-start',
            'Spike-start',
            'Spike',
            'C4P4T4T6尖波棘波',
            'C4尖波',
            'O1O2T5T6棘慢波放电'
            'F3T3T5棘慢波',
            '广泛性2-3Hz棘慢波',
            'O1T5多棘慢波',
            '全导多棘慢波',
            'T3T4F7棘慢波',
            'O2T6棘慢波',
            '广泛性2-3Hz多棘慢波',
            'T3T4F7FZ棘慢波',
            'F3T3C31-2Hz棘慢波',
            'FZ尖波',
            'O1T5棘慢波慢波',
            'O1多棘波',
            '多棘慢',
            'FZ棘慢波',
            'P4O2T61-2Hz棘慢波',
            'O2T4T6棘波',
            'T3尖慢波',
            'O2T6多棘波',
            'O1T6棘波',
            'O1棘波',
            'T5O1棘慢波',
            '广泛性3-4Hz高波幅棘',
            'C4P4T4T6多棘波',
            '脑电图2-3HZ棘慢波',
            '全导棘波节律',
            'FZF3棘慢波',
            'T3FZ尖慢波',
            'T4T6尖波',
            'O1T5尖波',
            '左侧FP1F3尖波节律起',
            '全导3-4Hz高波幅棘慢',
            '全导3-4Hz棘慢波放电',
            'F7T3尖波'
    ):
        return True
    '''

    if (s.startswith('!') or s.startswith('！')) and '尖波' in s:
        return True
    if '慢波' in s:
        return False
    if s.startswith('？'):
        return False
    if '尖波' in s:
        return True
    if '左侧！' in s:
        return True
    if '~！' in s:
        return True
    if s.startswith('!') or s.startswith('！') or s.startswith('1'):
        return True
    if s == 's':
        return True
    if s == 'S':
        return True

    return False


class RawNihonReader():

    def __init__(self, fn, enhance_new_version=True):
        # if enhance_new_version:

        self.eeg_fn = Path(fn)

        # PNT/pnt compatible
        enhanced_nihon.enhance_nihon_pnt_compatible()

        if enhance_new_version:
            enhanced_nihon.enhance_nihon()
            logger.info('enhanced_nihon turn on')

        mne.io.nihon.nihon._valid_headers.append('EEG-1200A V01.00')

        self.raw = mne.io.nihon.read_raw_nihon(fn, preload=True)

    def get_start_and_end(self):

        start_and_end_time_list = []

        start_end_text_list = []

        labels_list = self.get_label()

        for item in labels_list:
            text = item['text']
            text = text.strip()

            if 'start' in text:
                start_end_text_list.append(item)

            elif 'end' in text:
                start_end_text_list.append(item)

        # signal point
        start_point = 0

        index_len = len(start_end_text_list) - 1

        while start_point < index_len:
            start_and_end_dic = {}
            current_element = start_end_text_list[start_point]
            text = current_element['text']
            if 'start' not in text:
                start_point = start_point + 1
            elif 'start' in text:

                current_element = start_end_text_list[start_point]
                start_and_end_dic['start_tm'] = current_element['tm']
                end_num = 1
                flag = True
                while flag:

                    if 'start' not in (start_end_text_list[start_point + end_num]['text']):
                        end_num = end_num + 1
                        if start_point + end_num > index_len:
                            flag = False
                    else:
                        flag = False

                start_and_end_dic['end_tm'] = start_end_text_list[start_point + end_num - 1]['tm']
                start_point = start_point + end_num

                start_and_end_time_list.append(start_and_end_dic)

        return start_and_end_time_list

    def get_sleep_time(self):

        sleep_first_ide = {}
        sleep_last_ide = {}
        sleep_stages_start_and_end_time_list = []
        sleep_list = []

        labels_list = self.get_label()

        for item in labels_list:
            text = item['text']
            text = text.strip()
            if 'Waking' in text:
                sleep_list.append(item)

            elif 'Stage' in text:
                sleep_list.append(item)

        for i in range(len(sleep_list) - 1):
            time_flame_dic = {}
            current_element = sleep_list[i]
            next_element = sleep_list[i + 1]
            current_text = current_element['text']

            if 'Waking' in current_text:
                time_flame_dic["label"] = 0
            elif 'Stage' in current_text:
                if "1" in current_text:
                    time_flame_dic["label"] = 1
                elif "2" in current_text:
                    time_flame_dic["label"] = 2
                elif "3" in current_text:
                    time_flame_dic["label"] = 3
            else:
                print("not sleep label", current_text)
                time_flame_dic["label"] = -1

            time_flame_dic["start_tm"] = current_element['tm']
            time_flame_dic["end_tm"] = next_element['tm']

            sleep_stages_start_and_end_time_list.append(time_flame_dic)

            if i == 0:
                sleep_first_ide["label"] = time_flame_dic["label"]
                sleep_first_ide["tm"] = time_flame_dic["start_tm"]

            if i == (len(sleep_list) - 2):
                sleep_last_ide["label"] = time_flame_dic["label"]
                sleep_last_ide["tm"] = time_flame_dic["end_tm"]

        return sleep_stages_start_and_end_time_list, sleep_first_ide, sleep_last_ide

    def get_label(self):

        cmt_label = self._read_anno_by_cmt()
        log_label = self._read_anno_by_log()

        filtered_cmt_tags = []

        if not cmt_label:
            for cmt_item in cmt_label:
                filtered_cmt_tags.append(cmt_item)

            filtered_tags = filtered_cmt_tags + log_label
        else:
            filtered_tags = log_label
        return filtered_tags

    def _read_anno_by_cmt(self):
        fname = self.eeg_fn.with_suffix('.CMT')
        if not fname.exists():
            fname = self.eeg_fn.with_suffix('.cmt')
            if not fname.exists():
                return self
        try:
            with open(fname, 'rb') as fid:
                fid.seek(1045)
                n_annos = np.fromfile(fid, np.uint8, 1)[0]
                annos = []
                for t_block in range(n_annos):
                    saddr = 1066 + t_block * 560
                    tm_saddr = saddr + 3
                    log_saddr = saddr + 47
                    fid.seek(tm_saddr)
                    tm = np.fromfile(fid, '|S20', 1).astype('U20')[0]
                    fid.seek(log_saddr)
                    tlog = np.fromfile(fid, '|S500', 1)[0].decode('gbk')
                    time = datetime.strptime(tm, '%Y%m%d%H%M%S%f')
                    annos.append({'index': t_block, 'tm': time, 'text': tlog})
            if n_annos != len(annos):
                print(
                    f'[!] {fname}, n_annos!=len(annos): {n_annos}!={len(annos)}'
                )
            self.annos = annos

        except Exception as e:
            traceback.print_exc()

        return self.annos

    def _read_anno_by_log(self):

        annos = []
        log_fn = self.eeg_fn.with_suffix('.LOG')
        print("===log_fn===", log_fn)
        events = parseSubLOG(log_fn)

        for k, item in enumerate(events):
            time_str1 = item.Clock_time.decode('gb2312', errors='ignore').strip('\x00')
            time_str1 = time_str1.replace('(', '').replace(')', '')
            time1 = datetime.strptime(time_str1, '%y%m%d%H%M%S')

            subtime = item.sub_event_name.Clock_time_cccuuu.decode('gb2312', errors='ignore').strip('\x00')

            description = item.First_event.decode('gb2312', errors='ignore').strip('\x00')
            description2 = item.sub_event_name.Second_event.decode('gb2312', errors='ignore').strip('\x00')

            description = description + description2

            time_stamp = time_str1 + subtime
            time_stamp = datetime.strptime(time_stamp, '%y%m%d%H%M%S%f')

            if is_abnormal_label(description):
                annos.append({'index': k, 'tm': time_stamp, 'text': description})

        return annos

    def get_header(self):
        return self.raw._header


class EEGReader():

    def __init__(self, fn, chan_config):

        self.fn = fn
        self.chan_config = chan_config
        self.all_need_leads = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4',
                               'T5', 'T6', 'Fz', 'Cz', 'Pz', 'PG1', 'PG2', 'A1', 'A2',
                               'EKG1', 'EKG2', 'EMG1',
                               'EMG2', 'EMG3', 'EMG4']

        self.temporal_region_leads = ['F9', 'F10', 'T9', 'T10', 'P9', 'P10']

        self.mapping = {}

        self.label_style_dic = {}

        self.reader = RawNihonReader(fn)

        self.eeg_raw = self.reader.raw
        self.check_t_region()
        self.sample_rate = self.eeg_raw.info['sfreq']

        RESAMPLE_RATE = 500
        if self.sample_rate != RESAMPLE_RATE:

            raw_sample_rate = self.sample_rate
            logger.warning('Sample rate != %s is not supported !!! Resample as 500Hz' % (RESAMPLE_RATE))

            self.eeg_raw = self.eeg_raw.resample(500, npad="auto")

            self.eeg_raw._header['sfreq'] = RESAMPLE_RATE
            self.eeg_raw._header['n_samples'] = int(self.eeg_raw._header['n_samples'] / raw_sample_rate * RESAMPLE_RATE)

            for i in range(self.eeg_raw._header['n_ctlblocks']):
                for j in range(self.eeg_raw._header['controlblocks'][i]['n_datablocks']):
                    self.eeg_raw._header['controlblocks'][i]['datablocks'][j]['sfreq'] = RESAMPLE_RATE
                    self.eeg_raw._header['controlblocks'][i]['datablocks'][j]['n_samples'] = \
                        int(self.eeg_raw._header['controlblocks'][i]['datablocks'][j][
                                'n_samples'] / raw_sample_rate * RESAMPLE_RATE)

            self.sample_rate = self.eeg_raw._header['sfreq']

        self.sample_rate = int(self.sample_rate)

        logger.info('Raw eeg info: %s', self.eeg_raw.info)

        self._do_lead_check_pick_reorder()

        logger.info('Using channel mapp %s', self.mapping)
        # filter eeg signal

        # 所有通道
        fifty_hz_filter_channels = self.have_channels
        self.eeg_raw.notch_filter([50], picks=fifty_hz_filter_channels)

        # 脑电
        eeg_channels = [i for i in self.all_need_leads[:-6]]
        eeg_channels = [x for x in eeg_channels if x in self.have_channels]
        self.eeg_raw.filter(l_freq=0.1, h_freq=70, fir_design='firwin', picks=eeg_channels)

        # 心电
        ecg_channels = self.all_need_leads[-6:-4]
        ecg_channels = [x for x in ecg_channels if x in self.have_channels]
        self.eeg_raw.filter(l_freq=0.05, h_freq=150, fir_design='firwin', picks=ecg_channels)

        # 肌电
        emg_channels = self.all_need_leads[-4:]
        emg_channels = [x for x in emg_channels if x in self.have_channels]
        self.eeg_raw.filter(l_freq=20, h_freq=None, fir_design='firwin', picks=emg_channels)

        self._set_fixed_data()

        self.blocks = self.get_blocks()

        self.annotations = self.reader.get_label()

        self.start_and_end = self.reader.get_start_and_end()

        self.sleep_time = self.reader.get_sleep_time()

        # 5 seconds
        self.slice_length_seconds = 4

        self.overlap_seconds = 0

        self.boundary_seconds = 0.1

    def check_t_region(self):

        self.with_temporal = False

        for ch in self.temporal_region_leads:
            if ch in self.eeg_raw.ch_names:
                self.with_temporal = True

    def _do_lead_check_pick_reorder(self, ):

        if os.access(self.chan_config, os.F_OK):

            # if with mapping file do characters mapping, eg. FP1->Fp1
            self._character_mapping()

            extra_mapping = json.load(open(self.chan_config, mode='r'))

            # reverse key and value, because key is the target
            extra_mapping = {v: k for k, v in extra_mapping.items()}
            for k, v in extra_mapping.items():
                if k in self.eeg_raw.ch_names:
                    self.mapping[k] = v
            logger.info('Map channels with %s' % (self.chan_config))
        else:
            logger.info('Map channels with function default')

            # do xiehe mapping
            self._character_mapping()
            self._base_mapping()

            if 'out' in self.fn:
                print(self.fn + ' using outpatient version')
                self._outp_mapping()

            else:
                print(self.fn + ' using inpatient version')
                self._inp_mapping()

        self.eeg_raw = self.eeg_raw.rename_channels(self.mapping)

        # pick
        self.have_channels = []
        self.missing_channels = []

        for i, ch in enumerate(self.all_need_leads):
            if ch in self.eeg_raw.ch_names:
                self.have_channels.append(ch)
            else:
                self.missing_channels.append(ch)

        if len(self.missing_channels) > 0:
            logger.warning('Some channels are missing, missing channels-%s', self.missing_channels)
        else:
            logger.warning('No channels are missing, missing channels-%s', self.missing_channels)

    def _set_fixed_data(self):
        self.fixed_data = np.zeros(shape=(len(self.all_need_leads), self.eeg_raw.n_times))

        missing_channels = []
        for i, ch in enumerate(self.all_need_leads):
            if ch in self.eeg_raw.ch_names:
                self.fixed_data[i, ...] = self.eeg_raw.get_data(ch)
            else:
                missing_channels.append(ch)

    def _character_mapping(self):
        mapping_dul = ''
        if "FZ" in self.eeg_raw.info['ch_names']:
            if "Fz" in self.eeg_raw.info['ch_names']:
                mapping_dul += 'Fz,FZ\t'
            else:
                self.mapping["FZ"] = "Fz"
        if "CZ" in self.eeg_raw.info['ch_names']:
            if "Cz" in self.eeg_raw.info['ch_names']:
                mapping_dul += 'Cz,CZ\t'
            else:
                self.mapping["CZ"] = "Cz"
        if "PZ" in self.eeg_raw.info['ch_names']:
            if "Pz" in self.eeg_raw.info['ch_names']:
                mapping_dul += 'Pz,PZ\t'
            else:
                self.mapping["PZ"] = "Pz"
        #
        if "FP1" in self.eeg_raw.info['ch_names']:
            if "Fp1" in self.eeg_raw.info['ch_names']:
                mapping_dul += 'Fp1,FP1\t'
            else:
                self.mapping["FP1"] = "Fp1"
        if "FP2" in self.eeg_raw.info['ch_names']:
            if "Fp2" in self.eeg_raw.info['ch_names']:
                mapping_dul += 'Fp2,FP2\t'
            else:
                self.mapping["FP2"] = "Fp2"

    def _base_mapping(self):
        try:
            mapping_dul = ''
            # eeg

            # 64 lead new map T7,T8,P7,P8,Z1,Z2

            if "T7" in self.eeg_raw.info['ch_names']:
                if "T3" in self.eeg_raw.info['ch_names']:
                    mapping_dul += 'T7,T3\t'
                else:
                    self.mapping["T7"] = "T3"
            if "T8" in self.eeg_raw.info['ch_names']:
                if "T4" in self.eeg_raw.info['ch_names']:
                    mapping_dul += 'T8,T4\t'
                else:
                    self.mapping["T8"] = "T4"
            if "P7" in self.eeg_raw.info['ch_names']:
                if "T5" in self.eeg_raw.info['ch_names']:
                    mapping_dul += 'P7,T5\t'
                else:
                    self.mapping["P7"] = "T5"
            if "P8" in self.eeg_raw.info['ch_names']:
                if "T6" in self.eeg_raw.info['ch_names']:
                    mapping_dul += 'P8,T6\t'
                else:
                    self.mapping["P8"] = "T6"
            if "Z1" in self.eeg_raw.info['ch_names']:
                if "PG1" in self.eeg_raw.info['ch_names']:
                    mapping_dul += 'Z1,PG1\t'
                else:
                    self.mapping["Z1"] = "PG1"
            if "Z2" in self.eeg_raw.info['ch_names']:
                if "PG2" in self.eeg_raw.info['ch_names']:
                    mapping_dul += 'Z2,PG2\t'
                else:
                    self.mapping["Z2"] = "PG2"

            if "PG1-0" in self.eeg_raw.info['ch_names']:
                if "PG1" in self.eeg_raw.info['ch_names']:
                    mapping_dul += 'PG1,PG1-0\t'
                else:
                    self.mapping["PG1-0"] = "PG1"
            if "PG1-1" in self.eeg_raw.info['ch_names']:
                if "PG2" in self.eeg_raw.info['ch_names']:
                    mapping_dul += 'PG2,PG1-1\t'
                else:
                    self.mapping["PG1-1"] = "PG2"

            # ekg
            if "T1" in self.eeg_raw.info['ch_names']:
                if "EKG1" in self.eeg_raw.info['ch_names']:
                    mapping_dul += 'EKG1,T1\t'
                else:
                    self.mapping["T1"] = "EKG1"
            if "T2" in self.eeg_raw.info['ch_names']:
                if "EKG2" in self.eeg_raw.info['ch_names']:
                    mapping_dul += 'EKG2,T2\t'
                else:
                    self.mapping["T2"] = "EKG2"

            # emg [use it anyway]
            if "X1" in self.eeg_raw.info['ch_names']:
                if "EMG1" in self.eeg_raw.info['ch_names']:
                    mapping_dul += 'EMG1,X1\t'
                else:
                    self.mapping["X1"] = "EMG1"
            if "X2" in self.eeg_raw.info['ch_names']:
                if "EMG2" in self.eeg_raw.info['ch_names']:
                    mapping_dul += 'EMG2,X2\t'
                else:
                    self.mapping["X2"] = "EMG2"
            if "X3" in self.eeg_raw.info['ch_names']:
                if "EMG3" in self.eeg_raw.info['ch_names']:
                    mapping_dul += 'EMG3,X3\t'
                else:
                    self.mapping["X3"] = "EMG3"
            if "X4" in self.eeg_raw.info['ch_names']:
                if "EMG4" in self.eeg_raw.info['ch_names']:
                    mapping_dul += 'EMG4,X4\t'
                else:
                    self.mapping["X4"] = "EMG4"

            # emg
            if 'EMG3-1' in self.eeg_raw.info['ch_names']:
                if "EMG3" in self.eeg_raw.info['ch_names']:
                    mapping_dul += 'EMG3,EMG3-1\t'
                else:
                    self.mapping["EMG3-1"] = "EMG3"
            if 'EMG4-1' in self.eeg_raw.info['ch_names']:
                if "EMG4" in self.eeg_raw.info['ch_names']:
                    mapping_dul += 'EMG4,EMG4-1\t'
                else:
                    self.mapping["EMG4-1"] = "EMG4"
            if "EMG3" not in self.mapping.values() and "EMG3-0" in self.eeg_raw.info['ch_names']:
                if "EMG3" in self.eeg_raw.info['ch_names']:
                    mapping_dul += 'EMG3,EMG3-0\t'
                else:
                    self.mapping["EMG3-0"] = "EMG3"
            if "EMG4" not in self.mapping.values() and "EMG4-0" in self.eeg_raw.info['ch_names']:
                if "EMG4" in self.eeg_raw.info['ch_names']:
                    mapping_dul += 'EMG4,EMG4-0\t'
                else:
                    self.mapping["EMG4-0"] = "EMG4"

        except Exception as e:
            traceback.print_exc()
            self.do_calc = False

    def _outp_mapping(self):
        try:
            mapping_dul = ''
            # print("self.mapping: ", self.mapping)
            print("self.eeg_raw.info: ", self.eeg_raw.info)

            # ekg
            if "A25" in self.eeg_raw.info['ch_names']:
                self.mapping["A25"] = "EKG1"
            if "A26" in self.eeg_raw.info['ch_names']:
                self.mapping["A26"] = "EKG2"

            valid2728 = 0
            valid3132 = 0
            if "A27" in self.eeg_raw.info['ch_names'] and "A28" in self.eeg_raw.info['ch_names']:
                i1, i2 = self.eeg_raw.info['ch_names'].index('A27'), self.eeg_raw.info['ch_names'].index('A28')
                valid2728 = any(self.eeg_raw[i1, :][0][0]) and any(
                    self.eeg_raw[i2, :][0][0]) and len(self.eeg_raw[i1, :][0][0]) == len(
                    self.eeg_raw[i2, :][0][0]) == len(self.eeg_raw[0, :][0][0])
                print("A27,A28 exists, and valid=", valid2728)
            if "A31" in self.eeg_raw.info['ch_names'] and "A32" in self.eeg_raw.info['ch_names']:
                i1, i2 = self.eeg_raw.info['ch_names'].index('A31'), self.eeg_raw.info['ch_names'].index('A32')
                valid3132 = any(self.eeg_raw[i1, :][0][0]) and any(
                    self.eeg_raw[i2, :][0][0]) and len(self.eeg_raw[i1, :][0][0]) == len(
                    self.eeg_raw[i2, :][0][0]) == len(self.eeg_raw[0, :][0][0])
                print("A31,A32 exists, and valid=", valid3132)
            if valid2728:
                self.mapping["A27"] = "EMG1"
                self.mapping["A28"] = "EMG2"
            elif valid3132:
                self.mapping["A31"] = "EMG1"
                self.mapping["A32"] = "EMG2"
            if "A29" in self.eeg_raw.info['ch_names']:
                self.mapping["A29"] = "EMG3"
            if "A30" in self.eeg_raw.info['ch_names']:
                self.mapping["A30"] = "EMG4"
            if mapping_dul:
                print('\nnot map as already exist: ' + mapping_dul)
            print("\noutp-mapping: ", self.mapping)

        except Exception as e:
            print(self.filename, "_outp_mapping error: ", e, "(eeg_ekg_emg_raw 1: ",
                  len(self.eeg_raw.info['ch_names']), self.eeg_raw.info['ch_names'], ")")
            self.do_calc = False

    def _inp_mapping(self):
        try:
            # print("self.mapping: ", self.mapping)
            print("self.eeg_raw.info: ", self.eeg_raw.info)
            mapping_dul = ''

            # ekg
            if "A24" in self.eeg_raw.info['ch_names']:
                if "EKG1" in self.eeg_raw.info['ch_names']:
                    mapping_dul += 'EKG1,A24\t'
                else:
                    self.mapping["A24"] = "EKG1"
            if "A25" in self.eeg_raw.info['ch_names']:
                if "EKG2" in self.eeg_raw.info['ch_names']:
                    mapping_dul += 'EKG2,A25\t'
                else:
                    self.mapping["A25"] = "EKG2"

            # emg
            if 'AEMG-0' in self.eeg_raw.info['ch_names']:
                if "EMG3" in self.eeg_raw.info['ch_names']:
                    mapping_dul += 'EMG3,AEMG-0\t'
                else:
                    self.mapping["AEMG-0"] = "EMG3"
            if 'AEMG-1' in self.eeg_raw.info['ch_names']:
                if "EMG4" in self.eeg_raw.info['ch_names']:
                    mapping_dul += 'EMG4,AEMG-1\t'
                else:
                    self.mapping["AEMG-1"] = "EMG4"
            if 'AEMG1' in self.eeg_raw.info['ch_names']:
                if "EMG3" in self.eeg_raw.info['ch_names']:
                    mapping_dul += 'EMG3,AEMG1\t'
                else:
                    self.mapping["AEMG1"] = "EMG3"
            if 'AEMG2' in self.eeg_raw.info['ch_names']:
                if "EMG4" in self.eeg_raw.info['ch_names']:
                    mapping_dul += 'EMG4,AEMG2\t'
                else:
                    self.mapping["AEMG2"] = "EMG4"
            if np.array([i.startswith("EOG1") or i.startswith('EOG2') for i in self.eeg_raw.info['ch_names']]).any():
                if "A28" in self.eeg_raw.info['ch_names']:
                    if "EMG1" in self.eeg_raw.info['ch_names']:
                        mapping_dul += 'EMG1,A28\t'
                    else:
                        self.mapping["A28"] = "EMG1"
                if "A29" in self.eeg_raw.info['ch_names']:
                    if "EMG2" in self.eeg_raw.info['ch_names']:
                        mapping_dul += 'EMG2,A29\t'
                    else:
                        self.mapping["A29"] = "EMG2"
                if "A30" in self.eeg_raw.info['ch_names']:
                    if "EMG3" in self.eeg_raw.info['ch_names']:
                        mapping_dul += 'EMG3,A30\t'
                    else:
                        self.mapping["A30"] = "EMG3"
                if "A31" in self.eeg_raw.info['ch_names']:
                    if "EMG4" in self.eeg_raw.info['ch_names']:
                        mapping_dul += 'EMG4,A31\t'
                    else:
                        self.mapping["A31"] = "EMG4"
            else:
                if "A26" in self.eeg_raw.info['ch_names']:
                    if "EMG1" in self.eeg_raw.info['ch_names']:
                        mapping_dul += 'EMG1,A26\t'
                    else:
                        self.mapping["A26"] = "EMG1"
                if "A27" in self.eeg_raw.info['ch_names']:
                    if "EMG2" in self.eeg_raw.info['ch_names']:
                        mapping_dul += 'EMG2,A27\t'
                    else:
                        self.mapping["A27"] = "EMG2"
                if "A28" in self.eeg_raw.info['ch_names']:
                    if "EMG3" in self.eeg_raw.info['ch_names']:
                        mapping_dul += 'EMG3,A28\t'
                    else:
                        self.mapping["A28"] = "EMG3"
                if "A29" in self.eeg_raw.info['ch_names']:
                    if "EMG4" in self.eeg_raw.info['ch_names']:
                        mapping_dul += 'EMG4,A29\t'
                    else:
                        self.mapping["A29"] = "EMG4"
            if mapping_dul:
                print('\nnot map as already exist: ' + mapping_dul)
            print("\ninp-mapping: ", self.mapping)

        except Exception as e:
            print(self.filename, "_inp_mapping error: ", e, "(eeg_ekg_emg_raw 1: ",
                  len(self.eeg_raw.info['ch_names']), self.eeg_raw.info['ch_names'], ")")
            self.do_calc = False

    def get_blocks(self, ):

        start_p = 0
        data_blocks = []
        try:
            for item in self.reader.raw._header['controlblocks'][0]['datablocks']:
                one_block = {}
                end_p = start_p + item["n_samples"]

                data = self.fixed_data[:, start_p:end_p]

                one_block['data'] = data
                one_block['start_time'] = item['start_time']
                one_block['start_point'] = start_p
                one_block['sfreq'] = item["sfreq"]
                one_block['n_samples'] = item["n_samples"]

                one_block['end_time'] = (datetime.strptime(item['start_time'], '%y%m%d%H%M%S') + \
                                         timedelta(seconds=item["n_samples"] / item["sfreq"])).strftime("%Y%m%d%H%M%S")
                one_block['end_point'] = end_p
                data_blocks.append(one_block)

                # next block start
                start_p = end_p
        except:
            traceback.print_exc()

        return data_blocks

    def slidding(self):

        data_blocks = self.blocks
        data_list = []
        for block in data_blocks:

            data = block['data']
            block_start_time = block['start_time']
            block_start_time = datetime.strptime(block_start_time, '%y%m%d%H%M%S')

            slice_length = self.slice_length_seconds * self.sample_rate

            jump = (self.slice_length_seconds - self.overlap_seconds) * self.sample_rate

            for i in range(0, data.shape[-1], jump):
                one_sample = {}
                one_sample['text'] = ''
                if i + slice_length < data.shape[-1]:
                    cur_slice = np.array(data[:, i:i + slice_length])
                else:
                    cur_slice = np.array(data[:, i:])

                one_sample['data'] = cur_slice
                one_sample['start_point'] = block['start_point'] + i
                one_sample['start_time'] = block_start_time + timedelta(seconds=i / self.sample_rate)

                c, l = cur_slice.shape

                one_sample['end_time'] = one_sample['start_time'] + timedelta(seconds=l / self.sample_rate)

                for ann in self.annotations:
                    tag_time = ann['tm']

                    if one_sample['start_time'] + timedelta(seconds=self.boundary_seconds) < tag_time < one_sample[
                        'end_time'] - timedelta(seconds=self.boundary_seconds):
                        one_sample['text'] = ann['text']
                        one_sample['tag_time'] = tag_time

                data_list.append(one_sample)

        return data_list


if __name__ == '__main__':

    fn = './DA001019/DA001019.EEG'

    try:
        reader = RawNihonReader(False)
        reader.read(fn)
        reader.read_anno()
        print('stime', reader.raw.times[0])

        start_p = 0
        for item in reader.raw.header['controlblocks'][0]['datablocks']:
            end_p = start_p + item["n_samples"]
            data = reader.raw.get_data(start=start_p, stop=end_p)

            print(data.shape)

            start_p = end_p
    except:
        traceback.print_exc()
