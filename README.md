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
To pre-train with PEMR on the sepcific dataset from a scratch:
```
python3 main.py --masked_factor [THE RATIO OF FRAME BEING MASKED] --dataset_dir [DIRECTORY OF PRE-TRAINING DATASET] --batch_size [] ......
```

## Linear Evaluation
Fix the pre-trained encoder: 
```
python3 finetuning.py --finetune 0 --checkpoint_path [PATH OF PRE-TRAINED MODEL FILE]
```
## Fine-tuning
The parameters of pre-trained encoder are updated:
```
python3 finetuning.py --finetune 1 --checkpoint_path [PATH OF PRE-TRAINED MODEL FILE]
```

## Note
Pre-training, linear_evaluation and fine-tuning are all need audio .wav files. If you need the .wav files of music, you can contact us through yaodongai@zju.edu.cn.

## Pre-trained Model File
We provide our [pre-trained model](https://github.com/yaoodng7/PEMR/releases/download/pre-trained/pre-trained_model.ckpt) file obtained by pre-training for 300 epochs with 48 batch size. More details about this pre-trained model file can be find in https://github.com/yaoodng7/PEMR/releases
  


## Detail about dataset folder
We provide the dataset split files of MagnaTagATune and SHS100K. The GTZAN is only for training and the covers80 is only for testing. All audio files of these datasets are too big. The readers can contact with us for audio files.
