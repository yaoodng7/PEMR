# Contrastive Learning with Positive-Negative Frame Mask for Music Representation

Dong Yao, Zhou Zhao, Shengyu Zhang, Jieming Zhu,Yudong Zhu, Rui Zhang, Xiuqiang He

WWW2022

The repository is the offical pytorch implementation of our WWW2022 paper.

## Abstract
Self-supervised learning, especially contrastive learning, has made an outstanding contribution to the development of many deep learning research fields. Recently, researchers in the acoustic signal processing field noticed its success and leveraged contrastive learn- ing for better music representation. Typically, existing approaches maximize the similarity between two distorted audio segments sampled from the same music. In other words, they ensure a seman- tic agreement at the music level. However, those coarse-grained methods neglect some inessential or noisy elements at the frame level, which may be detrimental to the model to learn the effective representation of music. Towards this end, this paper proposes a novel Positive-nEgative frame mask for Music Representation based on the contrastive learning framework, abbreviated as PEMR. Concretely, PEMR incorporates a Positive-Negative Mask Genera- tion module, which leverages transformer blocks to generate frame masks on Log-Mel spectrogram. We can generate self-augmented negative and positive samples by masking important components or inessential components, respectively. We devise a novel con- trastive learning objective to accommodate both self-augmented positives/negatives sampled from the same music. We conduct experiments on four public datasets. The experimental results of two music-related downstream tasks, music classification and cover song identification, demonstrate the generalization ability and trans- ferability of music representation learned by **PEMR**.


<img align="center" src="images/new_overview.pdf" style="  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 100%;" /> 


## Detail about dataset folder
We provide the dataset split files of MagnaTagATune and SHS100K. The GTZAN is only for training and the covers80 is only for testing. All audio files of these datasets are too big. The readers can contact with us for audio files.
