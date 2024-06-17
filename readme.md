## Our open dataset

We are delighted to introduce our open-source dataset, the Epileptic Spike Dataset, sourced from the Epilepsy Center of Peking Union Medical College Hospital (PUMCH). These invaluable resources are now available for research purposes, aimed at enhancing knowledge and fostering innovation in the realm of electroencephalography.

We have released 84 EDF files, including 29 channels, which include 19 channels according to 10-20 international standards, 4 ear pole channels, 2 electrocardiogram channels, and 4 electromyography channels.

```
international_10_20_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5','T6','Fz', 'Cz', 'Pz']
ear_channels = ['PG1', 'PG2', 'A1', 'A2']
heart_channels = ['EKG1', 'EKG2']
muscle_channels = ['EMG1','EMG2', 'EMG3', 'EMG4']
```

There are three types of discharge event identifiers in the comments of EDF:

```
{
'label1':  '!',
'label2':  '!start',
'label3':  '!end',
'label4':  'Waking',
'label5':  'Sleeping'
}
'!'：means that discharge occurs at this moment 
'!start': the start time of continuous discharge
'!end': the end time of continuous discharge
'Waking': During the period between the current label and the next label (label4, label5), it is in an awakened state.
'Sleeping': During the period between the current label and the next label (label4, label5), it remains in a sleeping state.
```

## Download EDF-Dataset

If you want to download the dataset, you can download data based on this link "[download.vepiset.com]()"

## Create a virtual environment with conda

```python
conda create -n vepiset python=3.9
conda activate vepiset
pip install -r requirements.txt
```

## Prepare dataset

1. EDF To Numpy Files

   ```
   python edf2npy.py --edf_files_path source_edf_data_path --numpy_files_path edf_to_npy_dir_path 
   ```

2. Generate CSV Files Based On The Numpy Files Path

   ```
   python numpy2csv.py --numpy_files_path eeg_to_npy_dir_path  --save_csv_file csv_file_path
   ```

## How to run

1.`vim train_config.py`

- `config.DATA.data_file`   

- `config.MODEL.model_path`

- `config.num_classes`  

**The setting of these two parameters is important**

2.`python train.py`

3.`python eval.py --test_path test_set_csv_file_path --weight model_file_path`

## Spike Data Set：

| Category   | IED    | non-IDE |
| ---------- | ------ | ------- |
| Quantity   | 2516   | 22933   |
| Percentage | 9.886% | 90.114% |

## 5 Cross Valid Result on Spike Data Set

|         | non-IDE | IDE  | PR    | RE    | F1    |
| ------- | ------- | ---- | ----- | ----- | ----- |
| non-IDE | 22549   | 384  | 0.976 | 0.983 | 0.980 |
| IDE     | 552     | 1964 | 0.836 | 0.781 | 0.808 |

