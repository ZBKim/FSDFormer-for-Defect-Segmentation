# FSDFormer

This repository contains the training and evaluation code for FSDFormer on surface defect segmentation datasets.

## News

- Dataset links, pretrained backbone checkpoints, and trained model checkpoints will be released soon.

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
```

Put the PVTv2 pretrained model into `pretrainModel`:

```text
pretrainModel/pvt_v2_b2.pth
```

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
