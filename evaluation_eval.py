import click
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['font.family'] = 'Times New Roman'
from ml.metrics import confusion_matrix, calculate_weighted_metrics, calculate_macro_metrics
from util.utils import ID_TO_ISCX_VPN, ID_TO_ISCX_NonVPN, ID_TO_ISCX_VPNNonVPN, ID_TO_ISCX_Tor, ID_TO_ISCX_NonTor, ID_TO_ISCX_TorNonTor

from ml.utils_1dcnn import (
    load_tor_traffic_classification_cnn1d_model,
    load_nontor_traffic_classification_cnn1d_model,
    load_tornontor_traffic_classification_cnn1d_model,
    load_application_classification_cnn1d_model,
    load_vpn_traffic_classification_cnn1d_model,
    load_nonvpn_traffic_classification_cnn1d_model,
    load_vpnnonvpn_traffic_classification_cnn1d_model,
    load_cic_classification_cnn1d_model,
    load_shenlan_classification_cnn1d_model,
)



model_task_map = {
    "cnn1d": {
        "app": load_application_classification_cnn1d_model,
        "tor": load_tor_traffic_classification_cnn1d_model,
        "nontor": load_nontor_traffic_classification_cnn1d_model,
        "tornontor": load_tornontor_traffic_classification_cnn1d_model,
        "vpn": load_vpn_traffic_classification_cnn1d_model,
        "nonvpn": load_nonvpn_traffic_classification_cnn1d_model,
        "vpnnonvpn": load_vpnnonvpn_traffic_classification_cnn1d_model,
        "cic": load_cic_classification_cnn1d_model,
        "shenlan": load_shenlan_classification_cnn1d_model
    },
}
class_task_map = {
    "vpn": ID_TO_ISCX_VPN,
    "nonvpn": ID_TO_ISCX_NonVPN,
    "vpnnonvpn": ID_TO_ISCX_VPNNonVPN,
    "tor": ID_TO_ISCX_Tor,
    "nontor": ID_TO_ISCX_NonTor,
    "tornontor": ID_TO_ISCX_TorNonTor
}

def plot_confusion_matrix(cm, labels, figname):
    labels = sorted(labels)
    normalised_cm = normalise_cm(cm)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        data=normalised_cm,
        # cmap='YlGnBu',
        cmap='Blues', cbar=True,
        xticklabels=labels, yticklabels=labels,
        annot_kws = {'size': 16, 'fontweight': 'normal', 'family': 'Times New Roman'},
        annot=True,
        ax=ax, fmt='.4f'
    )
    ax.set_xlabel('Predict Labels', fontsize=18)
    ax.set_ylabel('True Labels', fontsize=18)

    plt.xticks(rotation=30, ha='right', fontsize=18, family='Times New Roman')  # 旋转X轴标签
    plt.yticks(rotation=0, va='center', fontsize=18, family='Times New Roman')  # 旋转Y轴标签

    plt.tight_layout()
    plt.savefig(figname, bbox_inches='tight', dpi=300)
    plt.clf()
    
    

@click.command()
@click.option(
    "-d",
    "--data_path",
    help="test data dir path containing parquet files",
    required=True,
)
@click.option("-m", "--model_name", help="uesed model name, tscrnn or deeppacket", required=True)
@click.option("-l", "--model_path", help="loaded model path", required=True)
@click.option("-p", "--png_path", help="output confusion matrix png path", required=True)

@click.option(
    "-t",
    "--task",
    help='classification task. Option: "vpn", "nonvpn", "vpnnonvpn", "tor", "nontor" or "tornontor"',
    required=True,
)


def main(data_path, model_name, model_path, task, png_path):
    if model_name in model_task_map and task in model_task_map[model_name]:
        load_model = model_task_map[model_name][task](model_path, gpu=True)
    else:
        exit("Not Support")

    cm = confusion_matrix(data_path=data_path, model=load_model, num_class=len(class_task_map[task]))

    labels = []
    for i in sorted(list(class_task_map[task].keys())):
        labels.append(class_task_map[task][i])

    plot_confusion_matrix(cm, labels, png_path)
    print("Classification Weighted Metrics")
    accuracy, weighted_precision, weighted_recall, weighted_f1 = calculate_weighted_metrics(cm, labels)
    print(accuracy, weighted_precision, weighted_recall, weighted_f1)

    print("Classification Macro Metrics")
    accuracy, macro_precision, macro_recall, macro_f1 = calculate_macro_metrics(cm, labels)
    print(accuracy, macro_precision, macro_recall, macro_f1)


if __name__ == "__main__":
    main()
