from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import seed_everything
from ml.models.CNN1D import CNN1D

# from thop import profile

def train_CNN1D(
    data_path,
    epoch,
    model_path,
    logger,
    output_dim
):
    # prepare dir for model path
    if model_path:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

    # seed everything
    seed_everything(seed=9876, workers=True)

    model = CNN1D(
        data_path=data_path, output_dim=output_dim
    ).float()
    # input1 = torch.randn(32, 24, 24)
    # flops, params = profile(model, inputs=(input1,))
    # print('FLOPs = ' + str(flops / 1000 ** 2) + 'M')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')
    trainer = Trainer(
        devices=[0],
        log_every_n_steps=1,
        val_check_interval=1.0,
        max_epochs=epoch,
        # devices="auto",
        accelerator="auto",
        logger=logger,
        callbacks=[
            EarlyStopping(
                # monitor="training_loss", mode="min", patience=5, check_on_train_epoch_end=True
                monitor="val_loss", mode="min", patience=10, check_on_train_epoch_end=True
            )
        ],
    )
    trainer.fit(model)

    # save model
    trainer.save_checkpoint(str(model_path.absolute()))


def train_application_classification_cnn1d_model(data_path, model_path):
    logger = TensorBoardLogger(
        "application_classification_logs", "application_classification_cnn1d"
    )
    train_CNN1D(
        data_path=data_path,
        epoch=100,
        model_path=model_path,
        logger=logger,
        output_dim=17,
    )


def train_vpn_traffic_classification_cnn1d_model(data_path, model_path):
    logger = TensorBoardLogger(
        "vpn_traffic_classification_logs", "vpn_traffic_classification_cnn1d"
    )
    # print("train: ", data_path)
    train_CNN1D(
        data_path=data_path,
        epoch=100,
        model_path=model_path,
        logger=logger,
        output_dim=6,
    )


def train_nonvpn_traffic_classification_cnn1d_model(data_path, model_path):
    logger = TensorBoardLogger(
        "vpn_traffic_classification_logs", "nonvpn_traffic_classification_cnn1d"
    )
    # print("train: ", data_path)
    train_CNN1D(
        data_path=data_path,
        epoch=100,
        model_path=model_path,
        logger=logger,
        output_dim=6,
    )


def train_vpnnonvpn_traffic_classification_cnn1d_model(data_path, model_path):
    logger = TensorBoardLogger(
        "vpn_traffic_classification_logs", "vpnnonvpn_traffic_classification_cnn1d"
    )
    # print("train: ", data_path)
    train_CNN1D(
        data_path=data_path,
        epoch=100,
        model_path=model_path,
        logger=logger,
        output_dim=12,
    )


def train_tor_traffic_classification_cnn1d_model(data_path, model_path):
    logger = TensorBoardLogger(
        "tor_traffic_classification_logs", "tor_traffic_classification_cnn1d"
    )
    # print("train: ", data_path)
    train_CNN1D(
        data_path=data_path,
        epoch=100,
        model_path=model_path,
        logger=logger,
        output_dim=8,
    )


def train_nontor_traffic_classification_cnn1d_model(data_path, model_path):
    logger = TensorBoardLogger(
        "tor_traffic_classification_logs", "nontor_traffic_classification_cnn1d"
    )
    # print("train: ", data_path)
    train_CNN1D(
        data_path=data_path,
        epoch=100,
        model_path=model_path,
        logger=logger,
        output_dim=8,
    )


def train_tornontor_traffic_classification_cnn1d_model(data_path, model_path):
    logger = TensorBoardLogger(
        "tor_traffic_classification_logs", "tornontor_traffic_classification_cnn1d"
    )
    # print("train: ", data_path)
    train_CNN1D(
        data_path=data_path,
        epoch=100,
        model_path=model_path,
        logger=logger,
        output_dim=16,
    )


def train_shenlan_classification_cnn1d_model(data_path, model_path):
    logger = TensorBoardLogger(
        "shenlan_classification_logs", "shenlan_classification_cnn1d"
    )
    # print("train: ", data_path)
    train_CNN1D(
        data_path=data_path,
        epoch=80,
        model_path=model_path,
        logger=logger,
        output_dim=5,
    )


def train_cic_classification_cnn1d_model(data_path, model_path):
    logger = TensorBoardLogger(
        "shenlan_classification_logs", "shenlan_classification_cnn1d"
    )
    # print("train: ", data_path)
    train_CNN1D(
        data_path=data_path,
        epoch=80,
        model_path=model_path,
        logger=logger,
        output_dim=7,
    )


def load_model(model_path, gpu):
    if gpu:
        device = "cuda"
    else:
        device = "cpu"
    model = (
        CNN1D.load_from_checkpoint(
            str(Path(model_path).absolute()), map_location=torch.device(device)
        )
        .float()
        .to(device)
    )

    model.eval()

    return model


def load_application_classification_cnn1d_model(model_path, gpu=False):
    return load_model(model_path=model_path, gpu=gpu)


def load_vpn_traffic_classification_cnn1d_model(model_path, gpu=False):
    return load_model(model_path=model_path, gpu=gpu)


def load_nonvpn_traffic_classification_cnn1d_model(model_path, gpu=False):
    return load_model(model_path=model_path, gpu=gpu)


def load_vpnnonvpn_traffic_classification_cnn1d_model(model_path, gpu=False):
    return load_model(model_path=model_path, gpu=gpu)


def load_tor_traffic_classification_cnn1d_model(model_path, gpu=False):
    return load_model(model_path=model_path, gpu=gpu)


def load_nontor_traffic_classification_cnn1d_model(model_path, gpu=False):
    return load_model(model_path=model_path, gpu=gpu)


def load_tornontor_traffic_classification_cnn1d_model(model_path, gpu=False):
    return load_model(model_path=model_path, gpu=gpu)



def load_shenlan_classification_cnn1d_model(model_path, gpu=False):
    return load_model(model_path=model_path, gpu=gpu)


def load_cic_classification_cnn1d_model(model_path, gpu=False):
    return load_model(model_path=model_path, gpu=gpu)



def normalise_cm(cm):
    with np.errstate(all="ignore"):
        normalised_cm = cm / cm.sum(axis=1, keepdims=True)
        normalised_cm = np.nan_to_num(normalised_cm)
        return normalised_cm
