## Our open dataset

We are delighted to introduce our open-source dataset, the Epileptic Spike Dataset, sourced from the Epilepsy Center of Peking Union Medical College Hospital (PUMCH). These invaluable resources are now available for research purposes, aimed at enhancing knowledge and fostering innovation in the realm of electroencephalography.



We have released 84 MAT files, each including 29 electrodes: 19 standard 10-20 system electrodes, 4 auricular electrodes, 2 electrocardiogram (ECG) electrodes (LA and RA, under the clavicles), and 4 electromyogram (EMG) electrodes. Note: ECG channel is derived by ECG2 and ECG1 electrodes (ECG = ECG1(LA) - ECG2(RA)); EMG channel (left deltoid) is calculated from EMG2 and EMG1 (EMG(left) = EMG1 - EMG2), and EMG channel (right deltoid) is computed from EMG4 and EMG3 (EMG(right) = EMG3 - EMG4). These data processing steps can be found in the base_trainer/dataietr.py file.

```
international_10_20_electrodes = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5','T6','Fz', 'Cz', 'Pz']
ear_electrodes = ['PG1', 'PG2', 'A1', 'A2']
heart_electrodes = ['ECG1', 'ECG2']
muscle_electrodes = ['EMG1','EMG2', 'EMG3', 'EMG4']
29_electrodes_sequence = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5','T6','Fz', 'Cz', 'Pz','PG1', 'PG2', 'A1', 'A2','ECG1', 'ECG2','EMG1','EMG2', 'EMG3', 'EMG4']
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

## Download MAT-Dataset

If you want to download the dataset, you can download data based on this link (coming soon)

## Create a virtual environment with conda

```python
conda create -n vepiset python=3.9
conda activate vepiset
pip install -r requirements.txt
```

## Prepare dataset

1. MAT To Numpy Files

   ```
   python mat2npy.py --mat_files_path source_mat_data_path --numpy_files_path mat_to_npy_dir_path 
   ```
2. Generate CSV Files Based On The Numpy Files Path

   ```
   python numpy2csv.py --numpy_files_path mat_to_npy_dir_path  --save_csv_file csv_file_path
   ```

## How to run

1.`vim train_config.py`

- `config.DATA.data_file`
- `config.MODEL.model_path`

**The setting of these two parameters is important**

2.`python train.py`

3.`python eval.py --test_path test_set_csv_file_path --weight model_file_path`

## Spike Data Set：


| Category   | IED    | non-IED |
| ---------- | ------ | ------- |
| Quantity   | 2516   | 22933   |
| Percentage | 9.886% | 90.114% |

## 5 Cross Valid Result on Spike Data Set


|         | non-IED | IED  | PR    | RE    | F1    |
| ------- | ------- | ---- | ----- | ----- | ----- |
| non-IED | 22549   | 384  | 0.976 | 0.983 | 0.980 |
| IED     | 552     | 1964 | 0.836 | 0.781 | 0.808 |
