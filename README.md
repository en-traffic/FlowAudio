# FlowAudio
FlowAudio: Continuous Audio Representation for Encrypted Traffic Classification


## Environment Setup
Python 3.10.15
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia  #torch version: 2.5.1+cu124
pip install soundfile==0.12.1

## Pre-processing
Datasets: [ISCXVPN2016](https://www.unb.ca/cic/datasets/vpn.html) & [ISCXTOR2016](https://www.unb.ca/cic/datasets/tor.html)
Sessions: We use [SplitCap](https://www.netresec.com/?page=SplitCap) to obtain session for ISCX-VPN, ISCX-nonVPN, ISCX VPN-nonVPN, ISCX-Tor, ISCX-nonTor and ISCX Tor-nonTor datasets, respectively. 
Categorization Details: For specific categorization of each dataset, please refer to folder ISCX_data.

## Feature Extraction
