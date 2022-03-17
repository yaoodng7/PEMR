# Contrastive Learning with Positive-Negative Frame Mask for Music Representation

Dong Yao, Zhou Zhao, Shengyu Zhang, Jieming Zhu,Yudong Zhu, Rui Zhang, Xiuqiang He

WWW2022

[![arXiv](https://img.shields.io/badge/arXiv-2104.00305-b31b1b.svg)](https://arxiv.org/abs/2104.00305)

The repository is the offical pytorch implementation of our WWW2022 paper.

<div align="center">
  <img width="100%" alt="PEMR model" src="images/model.JPG?raw=true">
</div>
<div align="center">
  overview of PEMR
</div>

## Quick Start
```
git clone https://github.com/yaoodng7/PEMR.git

pip3 install -r requirements.txt
```

## Pre-training
```
python3 main.py --masked_factor [THE RATIO OF FRAME BEING MASKED] --dataset_dir [DIRECTORY OF PRE-TRAINING DATASET] --batch_size [] ......
```

## Linear Evaluation
```
python3 finetuning.py --finetune 0 --checkpoint_path [PATH OF PRE-TRAINED MODEL FILE]
```
## Fine-tuning
```
python3 finetuning.py --finetune 1 --checkpoint_path [PATH OF PRE-TRAINED MODEL FILE]
```
  
## Pre-trained
We provide our [pre-trained model file](https://github.com/yaoodng7/PEMR/releases/download/pre-trained_model.ckpt) obtained by pre-training for 300 epochs.
  


## Detail about dataset folder
We provide the dataset split files of MagnaTagATune and SHS100K. The GTZAN is only for training and the covers80 is only for testing. All audio files of these datasets are too big. The readers can contact with us for audio files.
