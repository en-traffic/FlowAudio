from pathlib import Path
import os
import click
import numpy as np
import torch
import binascii
import soundfile as sf
from joblib import Parallel, delayed

from util.utils import get_mfcc_from_wav, transfrom_bytes

def process_packet(packet):
    arr = bytes(packet)
    return arr

def read_pcap(path):
    with open(path, 'rb') as f:
        content = f.read()
    return content

def process_pcap(traffic_label, pcap_path, output_path, prefix):
    pcap_bytes = read_pcap(pcap_path)
    audio_data = transfrom_bytes(pcap_bytes)

    frame_rate = 16000  # 声音采样率
    wav_file_path = str(output_path.absolute() / (prefix + ".wav"))
    sf.write(wav_file_path, audio_data, frame_rate)
    feature_label = {}

    if os.path.exists(wav_file_path):
        mfcc_feature_original = get_mfcc_from_wav(wav_file_path) # (155, 24)
        os.remove(wav_file_path)
        mfcc_feature = mfcc_feature_original.transpose(0, 1).detach()

        feature_label = {
            "traffic_label": traffic_label,
            "mfcc_feature": mfcc_feature
        }

    rows = []
    if feature_label:
        rows.append(feature_label)

    return rows


def extract_pcap(path, output_path: Path = None):
    # 输出文件路径
    label = path.parent.name
    prefix = label + "." + path.name.rsplit(".", 1)[0].lower() # 文件名前缀

    print("Processing", path)
    traffic_label = int(path.parent.name)
    result= process_pcap(traffic_label, path, output_path, prefix)

    return result


@click.command()
@click.option(
    "-s",
    "--source",
    help="path to the directory containing raw pcap files",
    required=True,
)
@click.option(
    "-t",
    "--target",
    help="path to the directory for persisting preprocessed files",
    required=True,
)

@click.option("-n", "--njob", default=-1, help="num of executors", type=int)
def main(source, target, njob):
    data_dir_path = Path(source)
    target_dir_path = Path(target)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    data_list = []

    results = Parallel(n_jobs=njob)(
        delayed(extract_pcap)(
            pcap_path, target_dir_path
        )
        for pcap_path in sorted(data_dir_path.rglob('*.pcap*'))
    )
    for result in results:
        data_list.extend(result)

    print("Done")
    output_path = target_dir_path.absolute() / "dataset.pt"
    torch.save(data_list, output_path)
    print(f"Save at {output_path} ")

if __name__ == "__main__":
    main()
