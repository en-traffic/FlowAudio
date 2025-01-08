import numpy as np
import torch
import pandas as pd
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


def get_precision(cm, i):
    tp = cm[i, i]
    tp_fp = cm[:, i].sum()

    return tp / tp_fp


def get_recall(cm, i):
    tp = cm[i, i]
    p = cm[i, :].sum()

    return tp / p

def test_dataloader(data_path):
    data = torch.load(data_path, weights_only=True)
    X_test = data['X_test']
    y_test = data['y_test']
    X_test_tensor = torch.stack(X_test)
    y_test_tensor = torch.tensor(y_test)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=256, num_workers=8, shuffle=True)
    return test_loader

def test_dataloader_lstm(data_path):
    data = torch.load(data_path, weights_only=True)
    X_test = data['X_test']
    y_test = data['y_test']
    X_len_test = [len(x) for x in X_test]  # 计算序列长度
    X_test_tensor = torch.nn.utils.rnn.pad_sequence(X_test, batch_first=True)
    y_test_tensor = torch.tensor(y_test)
    test_dataset = TensorDataset(X_test_tensor, torch.tensor(X_len_test), y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=256, num_workers=8, shuffle=True)
    return test_loader


def confusion_matrix(data_path, model, num_class):
    # data_path = Path(data_path)
    model.eval()
    # input1 = torch.randn(32, 24, 24)
    # input1 = input1.to(model.device)
    # flops, params = profile(model, inputs=(input1,))
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')

    cm = np.zeros((num_class, num_class), dtype=np.float64)
    dataloader = test_dataloader(data_path)
    for batch in dataloader:
        first_element_shape = batch[0].shape
        # print("Shape of the first element:", first_element_shape)
        x = batch[0].to(model.device)
        y = batch[1]
        y_hat = torch.argmax(F.log_softmax(model(x), dim=1), dim=1)

        for i in range(len(y)):
            cm[y[i], y_hat[i]] += 1

    return cm


def get_classification_report(cm, labels=None):
    rows = []
    for i in range(cm.shape[0]):
        precision = get_precision(cm, i)
        recall = get_recall(cm, i)
        if labels:
            label = labels[i]
        else:
            label = i

        row = {"label": label, "precision": precision, "recall": recall}
        rows.append(row)

    return pd.DataFrame(rows)


def calculate_weighted_metrics(cm, labels=None):
    """
    输入：
        cm: 混淆矩阵 (num_classes, num_classes)

    输出：
        accuracy: 总准确率
        weighted_precision: 加权精确率
        weighted_recall: 加权召回率（与加权准确率相同）
        weighted_f1: 加权 F1 分数
    """
    num_classes = cm.shape[0]

    # 初始化精确率、召回率、F1分数数组
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)

    # 真实类别的样本数（每列的和）
    support = np.sum(cm, axis=1)

    # 计算总样本数
    total_samples = np.sum(support)

    # 计算总的准确率
    accuracy = np.trace(cm) / total_samples

    # 计算每个类别的精确率、召回率和 F1 分数
    for i in range(num_classes):
        precision[i] = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
        recall[i] = cm[i, i] / np.sum(cm[:, i]) if np.sum(cm[:, i]) > 0 else 0
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1[i] = 0

    # 计算加权精确率、召回率和 F1 分数
    weighted_precision = np.sum((support / total_samples) * precision)
    weighted_recall = np.sum((support / total_samples) * recall)
    weighted_f1 = np.sum((support / total_samples) * f1)

    return accuracy, weighted_precision, weighted_recall, weighted_f1


def calculate_macro_metrics(cm, labels=None):
    """
    输入：
        cm: 混淆矩阵 (num_classes, num_classes)

    输出：
        accuracy: 总准确率
        macro_precision: 宏平均精确率
        macro_recall: 宏平均召回率
        macro_f1: 宏平均 F1 分数
    """
    num_classes = cm.shape[0]

    # 初始化精确率、召回率、F1分数数组
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)

    # 真实类别的样本数（每列的和）
    support = np.sum(cm, axis=1)

    # 计算总样本数
    total_samples = np.sum(support)

    # 计算总的准确率
    accuracy = np.trace(cm) / total_samples

    # 计算每个类别的精确率、召回率和 F1 分数
    for i in range(num_classes):
        precision[i] = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
        recall[i] = cm[i, i] / np.sum(cm[:, i]) if np.sum(cm[:, i]) > 0 else 0
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1[i] = 0

    # 计算宏平均精确率、召回率和 F1 分数
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    return accuracy, macro_precision, macro_recall, macro_f1


