import numpy as np
import mne
import os
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import sys
import traceback
from tqdm import tqdm
import gc
import argparse


class EDFReader():
    def __init__(self, fn,choice_type='spike'):
        self.fn = fn
        self.raw = mne.io.read_raw_edf(self.fn)
        self.choice_type = choice_type
        self.start_and_end = self.get_start_and_end()
        self.slice_length_seconds = 4
        self.sample_rate = 500
        self.boundary_seconds = 0.1


    def get_annotation(self,):
        final_ann = []
        annotations = self.raw.annotations
        for index in range(len(annotations)):
            label = annotations.description[index]
            final_ann.append(
                {
                    'index': index,
                    'tm':annotations.onset[index],
                    'text': annotations.description[index]
                })
        return final_ann

    def slidding(self,):
        data = self.raw.get_data()
        slice_length = self.slice_length_seconds * self.sample_rate
        data_list = []
        start_point = 0
        annotations = self.get_annotation()
        for i in range(0, data.shape[-1], slice_length):
            one_sample = {}
            one_sample['text'] = ''
            if i + slice_length < data.shape[-1]:
                cur_slice = np.array(data[:, i:i + slice_length])
            else:
                cur_slice = np.array(data[:, i:])

            one_sample['data'] = cur_slice
            one_sample['start_point'] = start_point + i
            one_sample['start_time'] = (start_point + i) / self.sample_rate

            c, l = cur_slice.shape
            one_sample['end_time'] = one_sample['start_time'] + (l / self.sample_rate)
            for ann in annotations:
                tag_time = ann['tm']
                if one_sample['start_time'] + self.boundary_seconds < tag_time < one_sample[
                    'end_time'] - self.boundary_seconds:
                    one_sample['text'] = ann['text']
                    one_sample['tag_time'] = tag_time
            data_list.append(one_sample)
        return data_list

    def get_start_and_end(self):

        start_and_end_time_list = []

        start_end_text_list = []

        labels_list = self.get_annotation()

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





class EDFCONVERTER():
    '''
    EDF to numpy
    '''
    def __init__(self,choice_type='spike'):
        self.choice_type = choice_type
        self.fix_label = [
            {'id':'DA00102R','start_point':276000,'end_point':278000,'label':0},
            {'id':'DA00102R','start_point':352000,'end_point':354000,'label':0},
            {'id':'DA00100V','start_point':384000,'end_point':386000,'label':0},
            {'id':'DA00103E','start_point':430000,'end_point':432000,'label':0},
            {'id':'DA00103Q','start_point':316000,'end_point':318000,'label':1},
            {'id':'DA001031','start_point':110000,'end_point':112000,'label':0},
        ] ## 边界情况，纠正
    def find_edf_files(self,folder_path):
        edf_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.edf'):
                    edf_files.append(os.path.join(root, file))

        return edf_files
    def check_and_mkdir(self,dir):
        if not os.access(dir, os.F_OK):
            os.mkdir(dir)

    def init_process(self,):
        def handle_exception(exc_type, exc_value, exc_traceback):
            # 处理异常的代码
            print('ERR HAPPEND')
            print(exc_traceback)
        sys.excepthook = handle_exception

    def wrapper_func(self,edf_fn, save_dir):
        try:
            self.process_one(edf_fn, save_dir)
        except:
            traceback.print_exc()
            return 1

        return 0

    def within_time_frame(self,item, times):
        slice_start_time = item['start_time']
        slice_end_time = item['end_time']
        for time in times:
            start_tm = time['start_tm']
            end_tm = time['end_tm']
            if (start_tm < slice_start_time < end_tm) or (start_tm < slice_end_time < end_tm):
                return True

        return False

    def is_abnormal_label(self,s: str):
        s = s.strip()
        if s.startswith('!'):
            return True

        return False


    def process_one(self,edf_fn, save_dir):
        eeg_id = edf_fn.rsplit('\\', 1)[1].rsplit('.', 1)[0]
        try:
            edf_reader = EDFReader(edf_fn, self.choice_type)
        except:
            traceback.print_exc()
            print(edf_fn)
            return {}

        save_dir = os.path.join(save_dir, eeg_id)

        self.check_and_mkdir(save_dir)

        message_dic = {}
        message_dic["eeg_id"] = eeg_id
        slices = edf_reader.slidding()
        times = edf_reader.start_and_end
        print(edf_reader.get_annotation())
        print(len(edf_reader.get_annotation()))

        for item in tqdm(slices):
            is_positive_sample_slice = self.within_time_frame(item, times)
            tag_text = item['text']
            label = 0
            if (tag_text == '' and (not is_positive_sample_slice)) or (tag_text != '' and (not self.is_abnormal_label(item['text']))):
                label = 0
            elif (self.is_abnormal_label(tag_text)) or (tag_text == '' and is_positive_sample_slice):
                label = 1
            time = item['start_time']
            waves = item['data']
            position = item['start_point']
            # if position == 124000 or position == 242000 or position == 268000 or position == 288000:
            #     print("is_positive_sample_slice",is_positive_sample_slice)
            #     print("self.is_abnormal_label(tag_text)",self.is_abnormal_label(tag_text))
            #     print("tag_text",tag_text)
            #     print("item",item)


            for fix_label in self.fix_label:
                if fix_label['id'] == str(eeg_id) and (fix_label['start_point'] == position):
                    label = fix_label['label']

            waves = waves.astype(np.float32)
            c, l = waves.shape
            save_f = os.path.join(save_dir, '%s_%d_%d_%d__%d.npy' % (str(eeg_id), position, position + l, 500, label))
            np.save(save_f, waves)

        gc.collect()


    def get_data(self,edf_data_dir, output_dir):
        data_dir = edf_data_dir
        edf_files = self.find_edf_files(data_dir)
        self.check_and_mkdir(output_dir)
        n_thread = 1
        process_unit = partial(self.wrapper_func, save_dir=output_dir)

        with ProcessPoolExecutor(max_workers=n_thread, initializer=self.init_process) as executor:
            # 提交任务到进程池
            futures = [executor.submit(process_unit, item) for item in edf_files]

            print(futures)


def main():
    parser = argparse.ArgumentParser(description='start process data')
    parser.add_argument('--edf_files_path', dest='edf_files_path', type=str, default=None, \
                        help='the path of edf files')
    parser.add_argument('--numpy_files_path', dest='numpy_files_path', type=str, default=None, \
                        help='the path of save numpy files')

    args = parser.parse_args()
    edf_files_path = args.edf_files_path
    numpy_files_path = args.numpy_files_path
    edfconverter = EDFCONVERTER(choice_type='spike')
    edf_files_path = r'D:\opends\opensource-dataset\EDF'
    numpy_files_path = r'D:\opends\opensource-dataset-numpy'
    edfconverter.get_data(edf_files_path,numpy_files_path)


if __name__ == '__main__':
    # main()
    ### test_edf
    edf_fn = r'D:\opends\0506_edf\DA00100A.edf'
    edf_reader = EDFReader(edf_fn, choice_type='spike')
    print(edf_reader.get_annotation())
    print(edf_reader.raw.info)





