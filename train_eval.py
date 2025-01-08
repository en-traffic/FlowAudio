import click

from ml.utils_tscrnn import (
    train_application_classification_tscrnn_model,
    train_vpn_traffic_classification_tscrnn_model,
    train_tor_traffic_classification_tscrnn_model,
)
from ml.utils_deeppacket import (
    train_tor_traffic_classification_deeppacket_model,
    train_application_classification_deeppacket_model,
    train_vpn_traffic_classification_deeppacket_model,
)
from ml.utils_2dcnn import (
    train_tor_traffic_classification_cnn2d_model,
    train_application_classification_cnn2d_model,
    train_vpn_traffic_classification_cnn2d_model,
    train_shenlan_classification_cnn2d_model,
    train_cic_classification_cnn2d_model
)
from ml.utils_1dcnn import (
    train_tor_traffic_classification_cnn1d_model,
    train_nontor_traffic_classification_cnn1d_model,
    train_tornontor_traffic_classification_cnn1d_model,
    train_application_classification_cnn1d_model,
    train_vpn_traffic_classification_cnn1d_model,
    train_nonvpn_traffic_classification_cnn1d_model,
    train_vpnnonvpn_traffic_classification_cnn1d_model,
    train_shenlan_classification_cnn1d_model,
    train_cic_classification_cnn1d_model
)
from ml.utils_cnn2dlstm import (
    train_tor_traffic_classification_cnn2dlstm_model,
    train_application_classification_cnn2dlstm_model,
    train_vpn_traffic_classification_cnn2dlstm_model,
)
from ml.utils_cnn2dtransformer import (
    train_tor_traffic_classification_cnn2dtrans_model,
    train_application_classification_cnn2dtrans_model,
    train_vpn_traffic_classification_cnn2dtrans_model,
)
from ml.utils_2dcnn_784 import (
    train_tor_traffic_784_classification_cnn2d_model
)
from ml.utils_1dcnn_100 import (
    train_tor_traffic_classification_cnn1d100_model
)
from ml.utils_lstm1dcnn import (
    train_tor_traffic_classification_lstm1dcnn_model,
    train_nontor_traffic_classification_lstm1dcnn_model,
    train_tornontor_traffic_classification_lstm1dcnn_model
)

model_task_map = {
    "deeppacket": {
        "app": train_application_classification_deeppacket_model,
        "tortraffic": train_tor_traffic_classification_deeppacket_model,
        "vpntraffic": train_vpn_traffic_classification_deeppacket_model
    },
    "tscrnn": {
        "app": train_application_classification_tscrnn_model,
        "tortraffic": train_tor_traffic_classification_tscrnn_model,
        "vpntraffic": train_vpn_traffic_classification_tscrnn_model
    },
    "cnn2d": {
        "app": train_application_classification_cnn2d_model,
        "tortraffic": train_tor_traffic_classification_cnn2d_model,
        "vpntraffic": train_vpn_traffic_classification_cnn2d_model,
        "shenlan": train_shenlan_classification_cnn2d_model,
        "cic": train_cic_classification_cnn2d_model
    },
    "cnn1d": {
        "app": train_application_classification_cnn1d_model,
        "tor": train_tor_traffic_classification_cnn1d_model,
        "nontor": train_nontor_traffic_classification_cnn1d_model,
        "tornontor": train_tornontor_traffic_classification_cnn1d_model,
        "vpn": train_vpn_traffic_classification_cnn1d_model,
        "nonvpn": train_nonvpn_traffic_classification_cnn1d_model,
        "vpnnonvpn": train_vpnnonvpn_traffic_classification_cnn1d_model,
        "shenlan": train_shenlan_classification_cnn1d_model,
        "cic": train_cic_classification_cnn1d_model
    },
    "cnn1d_100":{"tortraffic": train_tor_traffic_classification_cnn1d100_model},
    "cnn2d_784":{"tortraffic": train_tor_traffic_784_classification_cnn2d_model},
    "lstm1dcnn":{
        "nontor": train_nontor_traffic_classification_lstm1dcnn_model,
        "tor": train_tor_traffic_classification_lstm1dcnn_model,
        "tornontor": train_tornontor_traffic_classification_lstm1dcnn_model,
    },
    "cnn2dlstm": {
        "app": train_application_classification_cnn2dlstm_model,
        "tortraffic": train_tor_traffic_classification_cnn2dlstm_model,
        "vpntraffic": train_vpn_traffic_classification_cnn2dlstm_model
    },
    "cnn2dtransformer": {
        "app": train_application_classification_cnn2dtrans_model,
        "tortraffic": train_tor_traffic_classification_cnn2dtrans_model,
        "vpntraffic": train_vpn_traffic_classification_cnn2dtrans_model
    }
}


@click.command()
@click.option(
    "-d",
    "--data_path",
    help="training data dir path containing parquet files",
    required=True,
)
@click.option("-m", "--model_name", help="uesed model name, tscrnn or deeppacket", required=True)
@click.option("-o", "--model_path", help="output model path", required=True)
@click.option(
    "-t",
    "--task",
    help='classification task. Option: "app", "traffic", "cic", "shenlan" ',
    required=True,
)


def main(data_path, model_name, model_path, task):
    if model_name in model_task_map and task in model_task_map[model_name]:
        model_task_map[model_name][task](data_path, model_path)
    else:
        exit("Not Support")


if __name__ == "__main__":
    main()
