# FlowAudio
FlowAudio: Continuous Audio Representation for Encrypted Traffic Classification



## Dataset
Datasets: [ISCXVPN2016](https://www.unb.ca/cic/datasets/vpn.html) & [ISCXTOR2016](https://www.unb.ca/cic/datasets/tor.html)

Sessions: We use [SplitCap](https://www.netresec.com/?page=SplitCap) to obtain session for ISCX-VPN, ISCX-nonVPN, ISCX VPN-nonVPN, ISCX-Tor, ISCX-nonTor and ISCX Tor-nonTor datasets, respectively. 

Categorization Details: For specific categorization of each dataset, please refer to folder **ISCX_data**.

Note: The session pcaps with same class_id should be placed in one directory named class_id. 

## Environment Setup
Python 3.10.15

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia  #torch version: 2.5.1+cu124

pip install soundfile==0.12.1

## Audio Generation and Feature Extraction
```
python preprocessing.py -s <session_dir_path> -t <feature_dir_path> -n <njob>
session_dir_path  --The directory path to store sessions
feature_dir_path  --The directory path to store features, the feature file is dataset.pt
njob  --num of executors, default -1

```


## Create Train and Test
```
python create_train_test_set.py -s <feature_dir_path> -t <dataset_dir_path>
feature_dir_path  --The directory path to store features
dataset_dir_path  --The directory path to store datasets for training and evaluation
```


## Train Model
```
python train_eval.py -d <train_set_path> -m cnn1d -t <task> -o <model_path>
train_set_path  --The train set path
task  -- Classfication tasks, such as vpn, nonvpn, vpnnonvpn, tor, nontor, tornontor
model_path  --The path to output model
```

## Evaluation

```
python evaluation_eval.py -d <test_set_path> -m cnn1d -l <model_path> -p <confusion_matrix_figure_path> -t <task>
test_set_path  --The test set path
model_path  --The path to load model
confusion_matrix_path  --The path to output confusion matrix figure
task  -- Classfication tasks, such as vpn, nonvpn, vpnnonvpn, tor, nontor, tornontor
```

## Result
Our data is stored in folder **History**.

