import gc
import os
import sys
import mne
import json
import traceback
import numpy as np

sys.path.append('.')

from functools import partial
from tqdm import tqdm
from data.eeg_reader_dataset import EEGReader
from concurrent.futures import ProcessPoolExecutor


root_dir = '/data/sliced_data_sample_dataset'

mne.set_log_level(verbose=False)


def find_eeg_files(folder_path):
    npy_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.EEG'):
                npy_files.append(os.path.join(root, file))

    return npy_files


def check_and_mkdir(dir):
    if not os.access(dir, os.F_OK):
        os.mkdir(dir)


check_and_mkdir(root_dir)

def within_time_frame(item, times):
    slice_start_time = item['start_time']
    slice_end_time = item['end_time']
    for time in times:
        start_tm = time['start_tm']
        end_tm = time['end_tm']
        if (start_tm < slice_start_time < end_tm) or (start_tm < slice_end_time < end_tm):
            return True

    return False


def is_abnormal_label(s: str):
    s = s.strip()
    if '！g' in s:
        return False
    if '!g' in s:
        return False
    if s.startswith('!') or s.startswith('！'):
        return True

    return False


def save_npy_message_to_json(npy_info, path):
    with open(path, "w") as json_file:
        json.dump(npy_info, json_file)


def init_process():
    def handle_exception(exc_type, exc_value, exc_traceback):
        # 处理异常的代码
        print('ERR HAPPEND')
        print(exc_traceback)

    sys.excepthook = handle_exception


def process_one(eeg_fn, save_dir):
    eeg_id = eeg_fn.rsplit('/', 1)[1].rsplit('.', 1)[0]

    try:
        reader = EEGReader(eeg_fn, "default")

    except:
        traceback.print_exc()
        print(eeg_fn)
        return {}

    if reader.sample_rate != 500:
        print(eeg_fn, 'sample rate err', reader.sample_rate)
        return {}

    save_dir = os.path.join(save_dir, eeg_id)

    check_and_mkdir(save_dir)

    extra_tag = 1 if reader.with_temporal else 0

    message_dic = {}
    message_dic["eeg_id"] = eeg_id
    message_dic["extra_tag"] = extra_tag

    slices = reader.slidding()

    times = reader.start_and_end

    for item in tqdm(slices):

        is_positive_sample_slice = within_time_frame(item, times)

        tag_text = item['text']

        label = 0

        if (tag_text == '' and (not is_positive_sample_slice)) or (tag_text != '' and (not is_abnormal_label(item['text']))):
            label = 0

        elif (is_abnormal_label(tag_text)) or (tag_text == '' and is_positive_sample_slice):
            label = 1

        time = item['start_time']
        waves = item['data']
        position = item['start_point']
        waves = waves.astype(np.float32)
        c, l = waves.shape

        save_f = os.path.join(save_dir, '%s_%d_%d_%d__%d.npy' % (str(eeg_id), position, position + l, 500, label))
        np.save(save_f, waves)

    gc.collect()




def wrapper_func(eeg_fn, save_dir):
    try:
        process_one(eeg_fn, save_dir)
    except:
        traceback.print_exc()
        return 1

    return 0


def get_data(item, train_tset_dir='test'):
    data_dir = item['data_dir']
    egg_files = find_eeg_files(data_dir)

    save_dir = os.path.join(root_dir, train_tset_dir)

    check_and_mkdir(save_dir)
    save_dir = os.path.join(save_dir, data_dir.rsplit('/', 1)[-1])

    check_and_mkdir(save_dir)

    n_thread = 1
    process_unit = partial(wrapper_func, save_dir=save_dir)

    with ProcessPoolExecutor(max_workers=n_thread, initializer=init_process) as executor:
        # 提交任务到进程池
        futures = [executor.submit(process_unit, item) for item in egg_files]

        print(futures)


def find_eeg_files(folder_path):
    npy_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.EEG'):
                npy_files.append(os.path.join(root, file))

    return npy_files


def main():
    TEST_DIRS = ["DATASET"]

    TEST_DIRS = [{'data_dir': os.path.join('/data', x)} for x in TEST_DIRS]

    for test_dir in TEST_DIRS:
        get_data(test_dir)


if __name__ == '__main__':
    main()