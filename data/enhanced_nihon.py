
import functools
from datetime import datetime, timezone

import numpy as np
from mne.io.nihon import nihon
from mne.io.nihon.nihon import _ensure_path, _read_21e_file, _encodings, _mult_cal_one
from mne.utils import warn, logger

_valid_headers = [
    "EEG-1100A V01.00",
    "EEG-1100B V01.00",
    "EEG-1100C V01.00",
    "QI-403A   V01.00",
    "QI-403A   V02.00",
    "EEG-2100  V01.00",
    "EEG-2100  V02.00",
    "DAE-2100D V01.30",
    "DAE-2100D V02.00",
    'EEG-1200A V01.00',
]

_encodings = ('utf-8', 'GBK', 'latin1')


def bcd_to_number(bcd):
    number = ''
    for i in bcd:
        number += '{:02X}'.format(i)
    pos = number.rfind("F")
    if pos == 8:
        return '0'
    return number[pos + 1:]


def _read_nihon_header(fname):
    fname = _ensure_path(fname)
    _chan_labels = _read_21e_file(fname)
    header = {}
    with open(fname, 'rb') as fid:
        version = np.fromfile(fid, '|S16', 1).astype('U16')[0]
        if version not in _valid_headers:
            raise ValueError('Not a valid Nihon Kohden EEG file ({})'.format(version))

        fid.seek(0x0081)
        control_block = np.fromfile(fid, '|S16', 1).astype('U16')[0]
        if control_block not in _valid_headers:
            raise ValueError('Not a valid Nihon Kohden EEG file (control block {})'.format(version))

        fid.seek(0x17fe)
        waveform_sign = np.fromfile(fid, np.uint8, 1)[0]
        if waveform_sign != 1:
            raise ValueError('Not a valid Nihon Kohden EEG file (waveform block)')
        header['version'] = version
        controlblocks = []
        if control_block != 'EEG-1200A V01.00':
            fid.seek(0x0091)
            n_ctlblocks = np.fromfile(fid, np.uint8, 1)[0]
            header['n_ctlblocks'] = n_ctlblocks
            for i_ctl_block in range(n_ctlblocks):
                t_controlblock = {}
                fid.seek(0x0092 + i_ctl_block * 20)
                t_ctl_address = np.fromfile(fid, np.uint32, 1)[0]
                t_controlblock['address'] = t_ctl_address
                fid.seek(t_ctl_address + 17)
                n_datablocks = np.fromfile(fid, np.uint8, 1)[0]
                t_controlblock['n_datablocks'] = n_datablocks
                t_controlblock['datablocks'] = []
                for i_data_block in range(n_datablocks):
                    t_datablock = {}
                    fid.seek(t_ctl_address + i_data_block * 20 + 18)
                    t_data_address = np.fromfile(fid, np.uint32, 1)[0]
                    t_datablock['address'] = t_data_address
                    fid.seek(t_data_address + 20)
                    time = fid.read(6)
                    t_datablock['start_time'] = bcd_to_number(time)
                    fid.seek(t_data_address + 0x26)
                    t_n_channels = np.fromfile(fid, np.uint8, 1)[0]
                    t_datablock['n_channels'] = t_n_channels
                    t_datablock['wave_address'] = t_data_address + 0x27

                    t_channels = []
                    for i_ch in range(t_n_channels):
                        fid.seek(t_data_address + 0x27 + (i_ch * 10))
                        t_idx = np.fromfile(fid, np.uint8, 1)[0]
                        t_channels.append(_chan_labels[t_idx])

                    t_datablock['channels'] = t_channels
                    fid.seek(t_data_address + 0x1C)
                    t_record_duration = np.fromfile(fid, np.uint32, 1)[0]
                    t_datablock['duration'] = t_record_duration

                    fid.seek(t_data_address + 0x1a)
                    sfreq = np.fromfile(fid, np.uint16, 1)[0] & 0x3FFF
                    t_datablock['sfreq'] = sfreq

                    t_datablock['n_samples'] = int(t_record_duration * sfreq / 10)
                    t_controlblock['datablocks'].append(t_datablock)
                controlblocks.append(t_controlblock)
        else:
            # 得到控制块 E E G 1 ' 地址
            fid.seek(0x03EE)
            n_ctlblocks_address1 = int(np.fromfile(fid, np.uint64, 1)[0])
            fid.seek(n_ctlblocks_address1 + 17)
            n_ctlblocks1 = np.fromfile(fid, np.uint8, 1)[0]
            header['n_ctlblocks'] = n_ctlblocks1
            for i_ctl_block in range(n_ctlblocks1):
                fid.seek(n_ctlblocks_address1 + 18 + i_ctl_block * 24)
                t_ctl_address = int(np.fromfile(fid, np.uint64, 1)[0])
                fid.seek(t_ctl_address + 17)
                n_datablocks = np.fromfile(fid, np.uint16, 1)[0]
                t_controlblock = {'address': t_ctl_address, 'n_datablocks': n_datablocks, 'datablocks': []}
                for i_data_block in range(n_datablocks):
                    t_datablock = {}

                    fid.seek(t_ctl_address + 20 + i_data_block * 24)
                    t_data_address = int(np.fromfile(fid, np.uint64, 1)[0])
                    t_datablock['address'] = t_data_address

                    fid.seek(t_data_address + 22)
                    start_time = fid.read(12)
                    # print(start_time.decode('utf-8', errors='ignore'))
                    start_time = start_time.decode('utf-8', errors='ignore')
                    t_datablock['start_time'] = start_time

                    fid.seek(t_data_address + 68)
                    t_n_channels = np.fromfile(fid, np.uint32, 1)[0]
                    t_datablock['n_channels'] = t_n_channels

                    wave_addr = t_data_address + 72
                    t_datablock['wave_address'] = wave_addr

                    t_channels = []
                    for i_ch in range(t_n_channels):
                        fid.seek(wave_addr + (i_ch * 10))
                        t_idx = np.fromfile(fid, np.uint8, 1)[0]
                        t_channels.append(_chan_labels[t_idx])
                    t_datablock['channels'] = t_channels

                    fid.seek(t_data_address + 40)
                    sfreq = np.fromfile(fid, np.uint32, 1)[0]
                    t_datablock['sfreq'] = sfreq
                    t_record_duration = np.fromfile(fid, np.uint64, 1)[0]
                    t_datablock['duration'] = t_record_duration

                    t_datablock['n_samples'] = int(t_record_duration * sfreq / 10)
                    t_controlblock['datablocks'].append(t_datablock)
                controlblocks.append(t_controlblock)
        header['controlblocks'] = controlblocks

    # Now check that every data block has the same channels and sfreq
    chans = []
    sfreqs = []
    nsamples = []

    for t_ctl in header['controlblocks']:
        for t_dtb in t_ctl['datablocks']:
            chans.append(t_dtb['channels'])
            sfreqs.append(t_dtb['sfreq'])
            nsamples.append(t_dtb['n_samples'])
    for i_elem in range(1, len(chans)):
        if chans[0] != chans[i_elem]:
            raise ValueError('Channel names in datablocks do not match')
        if sfreqs[0] != sfreqs[i_elem]:
            raise ValueError('Sample frequency in datablocks do not match')
    header['ch_names'] = chans[0]
    header['sfreq'] = sfreqs[0]
    header['n_samples'] = np.sum(nsamples)

    # TODO: Support more than one controlblock and more than one datablock
    if header['n_ctlblocks'] != 1:
        raise NotImplementedError('I dont know how to read more than one control block for this type of file :(')
    if header['controlblocks'][0]['n_datablocks'] > 1:
        # Multiple blocks, check that they all have the same kind of data
        datablocks = header['controlblocks'][0]['datablocks']
        block_0 = datablocks[0]
        for t_block in datablocks[1:]:
            if block_0['n_channels'] != t_block['n_channels']:
                raise ValueError('Cannot read NK file with different number of channels in each datablock')
            if block_0['channels'] != t_block['channels']:
                raise ValueError('Cannot read NK file with different channels in each datablock')
            if block_0['sfreq'] != t_block['sfreq']:
                raise ValueError('Cannot read NK file with different sfreq in each datablock')

    return header


def _read_nihon_metadata(fname):
    metadata = {}
    fname = _ensure_path(fname)
    pnt_fname = fname.with_suffix(".PNT")
    if not pnt_fname.exists():
        warn("No PNT file exists. Try pnt file")

        pnt_fname = fname.with_suffix(".pnt")
        if not pnt_fname.exists():
            warn("No pnt file exists. Metadata will be blank")
            return metadata

    logger.info("Found PNT file, reading metadata.")
    with open(pnt_fname, "r") as fid:
        version = np.fromfile(fid, "|S16", 1).astype("U16")[0]
        if version not in _valid_headers:
            raise ValueError(f"Not a valid Nihon Kohden PNT file ({version})")
        metadata["version"] = version

        # Read timestamp
        fid.seek(0x40)
        meas_str = np.fromfile(fid, "|S14", 1).astype("U14")[0]
        meas_date = datetime.strptime(meas_str, "%Y%m%d%H%M%S")
        meas_date = meas_date.replace(tzinfo=timezone.utc)
        metadata["meas_date"] = meas_date

    return metadata


def _read_nihon_header_dev(fname, params):
    reserved = params['reserved']
    offset = params['offset']
    offset2 = params['offset2']

    fname = _ensure_path(fname)
    _chan_labels = _read_21e_file(fname)
    header = {}
    with open(fname, 'rb') as fid:
        version = np.fromfile(fid, '|S16', 1).astype('U16')[0]
        if version not in _valid_headers:
            raise ValueError('Not a valid Nihon Kohden EEG file ({})'.format(version))

        fid.seek(0x0081)
        control_block = np.fromfile(fid, '|S16', 1).astype('U16')[0]
        if control_block not in _valid_headers:
            raise ValueError('Not a valid Nihon Kohden EEG file (control block {})'.format(version))

        fid.seek(0x17fe)
        waveform_sign = np.fromfile(fid, np.uint8, 1)[0]
        if waveform_sign != 1:
            raise ValueError('Not a valid Nihon Kohden EEG file (waveform block)')
        header['version'] = version
        controlblocks = []
        if control_block != 'EEG-1200A V01.00':
            fid.seek(0x0091)
            n_ctlblocks = np.fromfile(fid, np.uint8, 1)[0]
            header['n_ctlblocks'] = n_ctlblocks
            for i_ctl_block in range(n_ctlblocks):
                t_controlblock = {}
                fid.seek(0x0092 + i_ctl_block * 20)
                t_ctl_address = np.fromfile(fid, np.uint32, 1)[0]
                t_controlblock['address'] = t_ctl_address
                fid.seek(t_ctl_address + 17)
                n_datablocks = np.fromfile(fid, np.uint8, 1)[0]
                t_controlblock['n_datablocks'] = n_datablocks
                t_controlblock['datablocks'] = []
                for i_data_block in range(n_datablocks):
                    t_datablock = {}
                    fid.seek(t_ctl_address + i_data_block * 20 + 18)
                    t_data_address = np.fromfile(fid, np.uint32, 1)[0]
                    t_datablock['address'] = t_data_address
                    fid.seek(t_data_address + 20)
                    time = fid.read(6)
                    t_datablock['start_time'] = bcd_to_number(time)
                    fid.seek(t_data_address + 0x26)
                    t_n_channels = np.fromfile(fid, np.uint8, 1)[0]
                    t_datablock['n_channels'] = t_n_channels
                    t_datablock['wave_address'] = t_data_address + 0x27

                    t_channels = []
                    for i_ch in range(t_n_channels):
                        fid.seek(t_data_address + 0x27 + (i_ch * 10))
                        t_idx = np.fromfile(fid, np.uint8, 1)[0]
                        t_channels.append(_chan_labels[t_idx])

                    t_datablock['channels'] = t_channels
                    fid.seek(t_data_address + 0x1C)
                    t_record_duration = np.fromfile(fid, np.uint32, 1)[0]
                    t_datablock['duration'] = t_record_duration

                    fid.seek(t_data_address + 0x1a)
                    sfreq = np.fromfile(fid, np.uint16, 1)[0] & 0x3FFF
                    t_datablock['sfreq'] = sfreq

                    t_datablock['n_samples'] = int(t_record_duration * sfreq / 10)
                    t_controlblock['datablocks'].append(t_datablock)
                controlblocks.append(t_controlblock)
        else:
            fid.seek(0x03EE)
            n_ctlblocks_address1 = int(np.fromfile(fid, np.uint64, 1)[0])
            fid.seek(n_ctlblocks_address1 + 17)
            n_ctlblocks1 = np.fromfile(fid, np.uint8, 1)[0]
            header['n_ctlblocks'] = n_ctlblocks1
            for i_ctl_block in range(n_ctlblocks1):
                t_controlblock = {}
                fid.seek(n_ctlblocks_address1 + 18 + i_ctl_block * 24)
                t_ctl_address = int(np.fromfile(fid, np.uint64, 1)[0])
                t_controlblock['address'] = t_ctl_address
                fid.seek(t_ctl_address)
                # id_datablocks = np.fromfile(fid, np.uint8, 1)[0]
                fid.seek(t_ctl_address + 17)
                n_datablocks = np.fromfile(fid, np.uint16, 1)[0]
                t_controlblock['n_datablocks'] = n_datablocks
                t_controlblock['datablocks'] = []
                for i_data_block in range(n_datablocks):
                    t_datablock = {}
                    fid.seek(t_ctl_address + 20 + i_data_block * 24)
                    t_data_address = int(np.fromfile(fid, np.uint64, 1)[0])
                    t_datablock['address'] = t_data_address
                    fid.seek(t_data_address + 20)
                    start_time = fid.read(20)
                    t_datablock['start_time'] = start_time
                    fid.seek(t_data_address + 60)
                    Reserved = int(np.fromfile(fid, np.uint16, 1)[0]) * reserved
                   # print(f'{i_data_block=}， {Reserved=}')
                    fid.seek(t_data_address + 62 + Reserved)
                    t_n_channels = np.fromfile(fid, np.uint32, 1)[0]
                    t_datablock['n_channels'] = t_n_channels
                    # t_datablock['wave_address'] = t_data_address + 66
                    t_datablock['wave_address'] = t_data_address + offset + Reserved + offset2

                    t_channels = []
                    for i_ch in range(t_n_channels):
                        fid.seek(t_data_address + offset + Reserved + (i_ch * 10))
                        t_idx = np.fromfile(fid, np.uint8, 1)[0]
                        t_channels.append(_chan_labels[t_idx])

                    t_datablock['channels'] = t_channels
                    fid.seek(t_data_address + 44)
                    t_record_duration = np.fromfile(fid, np.uint64, 1)[0]
                    t_datablock['duration'] = t_record_duration

                    fid.seek(t_data_address + 40)
                    sfreq = np.fromfile(fid, np.uint32, 1)[0]
                    t_datablock['sfreq'] = sfreq

                    t_datablock['n_samples'] = int(t_record_duration * sfreq / 10)
                    t_controlblock['datablocks'].append(t_datablock)
                controlblocks.append(t_controlblock)
        header['controlblocks'] = controlblocks

    # Now check that every data block has the same channels and sfreq
    chans = []
    sfreqs = []
    nsamples = []

    for t_ctl in header['controlblocks']:
        for t_dtb in t_ctl['datablocks']:
            chans.append(t_dtb['channels'])
            sfreqs.append(t_dtb['sfreq'])
            nsamples.append(t_dtb['n_samples'])
    for i_elem in range(1, len(chans)):
        if chans[0] != chans[i_elem]: raise ValueError('Channel names in datablocks do not match')
        if sfreqs[0] != sfreqs[i_elem]: raise ValueError('Sample frequency in datablocks do not match')
    header['ch_names'] = chans[0]
    header['sfreq'] = sfreqs[0]
    header['n_samples'] = np.sum(nsamples)

    # TODO: Support more than one controlblock and more than one datablock
    if header['n_ctlblocks'] != 1:
        raise NotImplementedError('I dont know how to read more than one control block for this type of file :(')
    if header['controlblocks'][0]['n_datablocks'] > 1:
        # Multiple blocks, check that they all have the same kind of data
        datablocks = header['controlblocks'][0]['datablocks']
        block_0 = datablocks[0]
        for t_block in datablocks[1:]:
            if block_0['n_channels'] != t_block['n_channels']:
                raise ValueError('Cannot read NK file with different number of channels in each datablock')
            if block_0['channels'] != t_block['channels']:
                raise ValueError('Cannot read NK file with different channels in each datablock')
            if block_0['sfreq'] != t_block['sfreq']:
                raise ValueError('Cannot read NK file with different sfreq in each datablock')

    return header


def _read_nihon_annotations(fname):
    fname = _ensure_path(fname)
    log_fname = fname.with_suffix('.LOG')
    if not log_fname.exists():
        log_fname = fname.with_suffix('.log')
        if not log_fname.exists():
            warn('No LOG file exists. Annotations will not be read')
            return dict(onset=[], duration=[], description=[])
    logger.info('Found LOG file, reading events.')
    with open(log_fname, 'r') as fid:
        version = np.fromfile(fid, '|S16', 1).astype('U16')[0]
        if version not in _valid_headers:
            raise ValueError('Not a valid Nihon Kohden LOG file ({})'.format(version))

        fid.seek(0x91)
        n_logblocks = np.fromfile(fid, np.uint8, 1)[0]
        all_onsets = []
        all_descriptions = []
        for t_block in range(n_logblocks):
            fid.seek(0x92 + t_block * 20)
            t_blk_address = np.fromfile(fid, np.uint32, 1)[0]
            fid.seek(t_blk_address + 0x12)
            n_logs = np.fromfile(fid, np.uint8, 1)[0]
            fid.seek(t_blk_address + 0x14)
            t_logs = np.fromfile(fid, '|S45', n_logs)
            for t_log in t_logs:
                for enc in _encodings:
                    try:
                        t_desc = t_log[:20].decode(enc).strip('\x00')
                    except UnicodeDecodeError:
                        pass
                    else:
                        break
                else:
                    warn(f"Could not decode t_desc log as one of {_encodings}")
                    continue
                for enc in _encodings:
                    try:
                        t_onset = t_log[20:26].decode(enc)
                        t_onset = datetime.strptime(t_onset, '%H%M%S')
                    except UnicodeDecodeError:
                        pass
                    else:
                        break
                else:
                    warn(f"Could not decode t_onset log as one of {_encodings}")
                    continue
                t_onset = (t_onset.hour * 3600 + t_onset.minute * 60 + t_onset.second)
                all_onsets.append(t_onset)
                all_descriptions.append(t_desc)

        annots = dict(onset=all_onsets, duration=[0] * len(all_onsets), description=all_descriptions)
    return annots


def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
    """Read a chunk of raw data."""
    # For now we assume one control block
    header = self._raw_extras[fi]["header"]

    # Get the original cal, offsets and gains
    cal = self._raw_extras[fi]["cal"]
    offsets = self._raw_extras[fi]["offsets"]
    gains = self._raw_extras[fi]["gains"]

    # get the right datablock
    datablocks = header["controlblocks"][0]["datablocks"]
    ends = np.cumsum([t["n_samples"] for t in datablocks])

    start_block = np.where(start < ends)[0][0]
    stop_block = np.where(stop <= ends)[0][0]

    if start_block != stop_block:
        # Recursive call for each block independently
        new_start = start
        sample_start = 0
        for t_block_idx in range(start_block, stop_block + 1):
            t_block = datablocks[t_block_idx]
            if t_block == stop_block:
                # If its the last block, we stop on the last sample to read
                new_stop = stop
            else:
                # Otherwise, stop on the last sample of the block
                new_stop = t_block["n_samples"] + new_start
            samples_to_read = new_stop - new_start
            sample_stop = sample_start + samples_to_read

            self._read_segment_file(
                data[:, sample_start:sample_stop],
                idx,
                fi,
                new_start,
                new_stop,
                cals,
                mult,
            )

            # Update variables for next loop
            sample_start = sample_stop
            new_start = new_stop
    else:
        datablock = datablocks[start_block]

        n_channels = datablock["n_channels"] + 1
        datastart = datablock["wave_address"] + (datablock["n_channels"] * 10)

        # Compute start offset based on the beginning of the block
        rel_start = start
        if start_block != 0:
            rel_start = start - ends[start_block - 1]
        start_offset = datastart + rel_start * n_channels * 2

        with open(self._filenames[fi], "rb") as fid:
            to_read = (stop - start) * n_channels
            fid.seek(start_offset)
            block_data = np.fromfile(fid, "<u2", to_read) + 0x8000
            block_data = block_data.astype(np.int16)
            block_data = block_data.reshape(n_channels, -1, order="F")
            block_data = block_data[:-1] *cal  # cast to float64
            block_data += offsets
            block_data *= gains
            _mult_cal_one(data, block_data, idx, cals, mult)

def _map_ch_to_specs(ch_name):
    unit_mult = 1e-6
    phys_min = -3200
    phys_max = 3199.902


    dig_min = -32768
    # if ch_name.upper() in _default_chan_labels:
    #     idx = _default_chan_labels.index(ch_name.upper())
    #     if (idx < 42 or idx > 73) and idx not in [76, 77]:
    #         unit_mult = 1e-6
    #         phys_min = -3200
    #         phys_max = 3199.902

    t_range = phys_max - phys_min
    cal = t_range / 65535
    offset = phys_min - (dig_min * cal)   # equal to phys_min-(phys_max - phys_min)/2

    out = dict(
        unit=unit_mult,
        phys_min=phys_min,
        phys_max=phys_max,
        dig_min=dig_min,
        cal=cal,
        offset=offset,
    )
    return out



use_enhanced = False
bak_valid_headers = nihon._valid_headers
bak_read_nihon_header = nihon._read_nihon_header
bak_read_nihon_annotations = nihon._read_nihon_annotations
bak_encodings = nihon._encodings
bak_read_segment_file = nihon.RawNihon._read_segment_file
bak_map_ch_to_specs=nihon._map_ch_to_specs


def enhance_nihon():
    global use_enhanced
    use_enhanced = True
    nihon._encodings = _encodings
    nihon._valid_headers = _valid_headers
    nihon._read_nihon_header = _read_nihon_header
    nihon._read_nihon_annotations = _read_nihon_annotations
    nihon.RawNihon._read_segment_file = _read_segment_file
    nihon._map_ch_to_specs=_map_ch_to_specs

def enhance_nihon_dev(params):
    global use_enhanced
    use_enhanced = True
    nihon._encodings = _encodings
    nihon._valid_headers = _valid_headers
    nihon._read_nihon_header = functools.partial(_read_nihon_header_dev, params=params)
    nihon._read_nihon_annotations = _read_nihon_annotations
    nihon.RawNihon._read_segment_file = _read_segment_file


def reset_nihon():
    global use_enhanced
    use_enhanced = False
    nihon._encodings = bak_encodings
    nihon._valid_headers = bak_valid_headers
    nihon._read_nihon_header = bak_read_nihon_header
    nihon._read_nihon_annotations = bak_read_nihon_annotations
    nihon.RawNihon._read_segment_file = bak_read_segment_file
    nihon._map_ch_to_specs = bak_map_ch_to_specs


def enhance_nihon_pnt_compatible():
    nihon._read_nihon_metadata=_read_nihon_metadata