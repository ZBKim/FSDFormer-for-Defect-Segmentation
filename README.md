# FSDFormer

This repository contains the training and evaluation code for FSDFormer on surface defect segmentation datasets.

## 
We provided the weights trained on three datasets：


[ZJU-Leaper](https://drive.google.com/file/d/1VUD2eXeNSBDo5U4yqKRpeAYqRLke6Fny/view?usp=sharing)
[CrackSeg9k](https://drive.google.com/file/d/1mTaJPNuagVMnmJZCMCLIdAokO345tz_O/view?usp=sharing)
[ESDIs-SOD](https://drive.google.com/file/d/1y5cSVnZBJ5Rq6pY3tsbYpP9p492xMVdn/view?usp=sharing)


## 1. Requirements

Recommended environment:

```text
python >= 3.7
pytorch
torchvision
timm
opencv-python
pillow
numpy
tqdm
tensorboard
albumentations
```

Install dependencies:

```bash
pip install torch torchvision torchaudio
pip install timm opencv-python pillow numpy tqdm tensorboard albumentations
```

## 2. Prepare Datasets and Pretrained Models

Create folders in the project root:

```bash
mkdir -p pretrainModel datasets
``
Download pre-trained weights[PVTv2-B2](https://drive.google.com/file/d/1aj0bt-8--K_zpLYVXcH252l__4kr1Cu1/view?usp=sharing), then put the PVTv2 pretrained model into `pretrainModel`:

```text
pretrainModel/pvt_v2_b2.pth
```

You can download the data set that we have already divided:
[CrackSeg9k](https://drive.google.com/file/d/1pOQBOjs_r9g6by0QQWU6hFT-dGeHlQqZ/view?usp=sharing)
[ZJU-Leaper](https://drive.google.com/file/d/1KpKTYP97BnsKvfB2PCZ_jiBrZQJwUnb8/view?usp=sharing)
[ESDI-SODs](https://drive.google.com/file/d/1KpKTYP97BnsKvfB2PCZ_jiBrZQJwUnb8/view?usp=sharing)


Put datasets into `datasets` with the following structure:

```text
datasets
├── CrackSeg9k
│   ├── train
│   │   ├── images
│   │   └── gt
│   └── test
│       ├── images
│       └── gt
├── ESDIs-SOD
│   ├── train
│   │   ├── images
│   │   └── gt
│   └── test
│       ├── images
│       └── gt
└── ZJU-Leaper
    ├── train
    │   ├── images
    │   └── gt
    └── test
        ├── images
        └── gt
```

Before training, modify the dataset root in `defect_train_multiGPU.py`:

```python
root = "./datasets"
```

Modify the PVTv2 pretrained model path in `model/FSDFormer.py`:

```python
path = "./pretrainModel/pvt_v2_b2.pth"
```

## 3. Train

Modify training settings in `defect_train_multiGPU.py`:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
dataset = "ESDIs-SOD"      # CrackSeg9k / ESDIs-SOD / ZJU-Leaper
bs = 32
lr = 1.6e-4
```

Run:

```bash
python defect_train_multiGPU.py
```

Training logs and checkpoints are saved in:

```text
ablationruns_esdis/
```

The best checkpoint is saved in:

```text
runs/{run_name}/checkpoints/
```

## 4. Test

Modify `defect_test.py`:

```python
from model.FSDFormer import FSDFormer
file_dir = "./datasets"
model_save = "./runs/{run_name}/checkpoints/{best_model}.pth"
```

Run:

```bash
python defect_test.py
```

Prediction maps are saved in:

```text
Pred/
```

## 5. Eval

Modify the prediction path and GT path in `eval.py`, then run:

```bash
python eval.py
```
