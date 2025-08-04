# DerProp 


## Getting Started

### Installation

```bash
git clone git@github.com:BBBBchan/CorrMatch.git
cd CorrMatch
conda create -n corrmatch python=3.9
conda activate corrmatch
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install opencv-python tqdm einops pyyaml
```

### Pretrained Backbone:
[ResNet-101](https://drive.google.com/file/d/1Rx0legsMolCWENpfvE2jUScT3ogalMO8/view?usp=sharing)
```bash
mkdir pretrained
```
Please put the pretrained model under `pretrained` dictionary.


### Dataset:

- Pascal VOC 2012: [JPEGImages](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) | [SegmentationClass](https://drive.google.com/file/d/1ikrDlsai5QSf2GiSUR3f8PZUzyTubcuF/view?usp=sharing)
- Cityscapes: [leftImg8bit](https://www.cityscapes-dataset.com/file-handling/?packageID=3) | [gtFine](https://drive.google.com/file/d/1E_27g9tuHm6baBqcA7jct_jqcGA89QPm/view?usp=sharing)

Please modify the dataset path in configuration files.*The groundtruth mask ids have already been pre-processed. You may use them directly.*

Your dataset path may look like:
```
├── [Your Pascal Path]
    ├── JPEGImages
    └── SegmentationClass
    
├── [Your Cityscapes Path]
    ├── leftImg8bit
    └── gtFine
```

## Usage

### Training

```bash
sh tools/train.sh <num_gpu> <port>
```
To run on different labeled data partitions or different datasets, please modify:

``config``, ``labeled_id_path``, ``unlabeled_id_path``, and ``save_path`` in train.sh.

### Evaluation
```bash
sh tools/val.sh <num_gpu> <port>
```
To evaluate your checkpoint, please modify ``checkpoint_path`` in val.sh.




## License
This code is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for non-commercial use only.
Please note that any commercial use of this code requires formal permission prior to use.

## Contact

For technical questions, please contact `yuanbinfu@tju.edu.cn`.


## Acknowledgement

We thank [CorrMatch](https://github.com/BBBBchan/CorrMatch/),  [UniMatch](https://github.com/LiheYoung/UniMatch), [CPS](https://github.com/charlesCXK/TorchSemiSeg), [CutMix-Seg](https://github.com/Britefury/cutmix-semisup-seg), [DeepLabv3Plus](https://github.com/YudeWang/deeplabv3plus-pytorch),  [U<sup>2</sup>PL](https://github.com/Haochen-Wang409/U2PL) and other excellent works (see this [project](https://github.com/BBBBchan/Awesome-Semi-Supervised-Semantic-Segmentation)) for their amazing projects!
