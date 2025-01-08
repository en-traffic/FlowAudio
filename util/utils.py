import torchaudio
import json
import numpy as np

ID_TO_ISCX_VPN_file = "ISCX_data/ID_TO_ISCX_VPN.json"
ID_TO_ISCX_NonVPN_file = "ISCX_data/ID_TO_ISCX_NonVPN.json"
ID_TO_ISCX_VPNNonVPN_file = "ISCX_data/ID_TO_ISCX_VPNNonVPN.json"
ID_TO_ISCX_Tor_file = "ISCX_data/ID_TO_ISCX_Tor.json"
ID_TO_ISCX_NonTor_file = "ISCX_data/ID_TO_ISCX_NonTor.json"
ID_TO_ISCX_TorNonTor_file = "ISCX_data/ID_TO_ISCX_TorNonTor.json"


def getClassName(class_path):
    with open(class_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


ID_TO_ISCX_VPN = getClassName(ID_TO_ISCX_VPN_file)
ID_TO_ISCX_NonVPN = getClassName(ID_TO_ISCX_NonVPN_file)
ID_TO_ISCX_VPNNonVPN = getClassName(ID_TO_ISCX_VPNNonVPN_file)
ID_TO_ISCX_Tor = getClassName(ID_TO_ISCX_Tor_file)
ID_TO_ISCX_NonTor = getClassName(ID_TO_ISCX_NonTor_file)
ID_TO_ISCX_TorNonTor =getClassName(ID_TO_ISCX_TorNonTor_file)


def transfrom_bytes(pcap_bytes):
    return np.array(pcap_bytes, dtype=np.int16)


def get_mfcc_from_wav(wav_file):
    waveform, sample_rate = torchaudio.load(wav_file)  # c读取音频文件
    frame_length = (25 / sample_rate) * 1000
    frame_shift = (10 / sample_rate) * 1000
    # 配置mfcc的参数
    mfcc_opts = {
        'num_mel_bins': 128, # Mel 滤波器数量
        'num_ceps': 24, # MFCC 特征向量的维度
        'channel': -1,
        'sample_frequency': sample_rate,  # 采样率，必须与音频文件的采样率匹配
        'frame_length': frame_length,
        'frame_shift': frame_shift,
        'window_type': 'hamming',
        # 'use_energy': False,
        'dither': 0
    }

    mfccs = torchaudio.compliance.kaldi.mfcc(waveform, **mfcc_opts)
    return mfccs

