from pathlib import Path
import click
import torch
from sklearn.model_selection import train_test_split
from collections import Counter


def load_pt_file(file_path):
    feature_label = torch.load(file_path, weights_only=True)
    return feature_label

def split_train_test(data, feature_col, label_col, test_size, data_path):
    features = [item[feature_col] for item in data]
    labels = [item[label_col] for item in data]

    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels, test_size=test_size, random_state=42
    )

    # 第二步：将临时集进一步划分为验证集和测试集（1:1）
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print("saving train datasets")
    data_path.mkdir(parents=True, exist_ok=True)
    torch.save({
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
    }, str(data_path/"train_dataset.pt"))

    print("saving test datasets")
    torch.save({
        'X_test': X_test,
        'y_test': y_test,
    }, str(data_path/"test_dataset.pt"))


def print_df_label_distribution(pt_path):
    train_path = Path(pt_path)/"train_dataset.pt"
    test_path = Path(pt_path)/"test_dataset.pt"
    train_data = torch.load(train_path, weights_only=True)
    test_data = torch.load(test_path, weights_only=True)

    train_labels = train_data['y_train']
    val_labels = train_data['y_val']
    test_labels = test_data['y_test']

    train_label_distribution = Counter(train_labels)
    test_label_distribution = Counter(test_labels)
    val_label_distribution = Counter(val_labels)

    print(f"Train label distribution: {train_label_distribution}")
    print(f"Test label distribution: {test_label_distribution}")
    print(f"Validation label distribution: {val_label_distribution}")


@click.command()
@click.option(
    "-s",
    "--source",
    help="path to the directory containing preprocessed files",
    required=True,
)
@click.option(
    "-t",
    "--target",
    help="path to the directory for persisting train and test set for both app and traffic classification",
    required=True,
)
@click.option("--test_size", default=0.2, help="size of test size", type=float)

def main(source, target, test_size):
    source_data_dir_path = Path(source)
    target_data_dir_path = Path(target)
    target_data_dir_path.mkdir(exist_ok=True)

    traffic_data_dir_path = target_data_dir_path / "traffic_classification"
    dataset_path = source_data_dir_path / "dataset.pt"
    data_list = load_pt_file(dataset_path)

    print("processing traffic classification dataset")
    split_train_test(
        data=data_list,
        feature_col="mfcc_feature",
        label_col="traffic_label",
        test_size=test_size,
        data_path=traffic_data_dir_path,
    )


    print("traffic label distribution")
    print_df_label_distribution(traffic_data_dir_path)



if __name__ == "__main__":
    main()
