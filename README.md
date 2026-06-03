# WPFormer

This repository contains the training and evaluation code for WPFormer on surface defect segmentation datasets.

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

Modify the PVTv2 pretrained model path in `model/WPFormer6.py`:

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
ablationruns_esdis/{run_name}/checkpoints/
```

## 4. Test

Modify `defect_test.py`:

```python
from model.WPFormer6 import WPFormer
file_dir = "./datasets"
model_save = "./ablationruns_esdis/{run_name}/checkpoints/{best_model}.pth"
```

Run:

```bash
python defect_test.py
```

Prediction maps are saved in:

```text
ablationPred/
```

## 5. Eval

Modify the prediction path and GT path in `eval.py`, then run:

```bash
python eval.py
```

## 6. Pretrained Models and Results

| Dataset | Dataset Download | Backbone | Input Size | Config | Checkpoint | Prediction Maps |
| --- | --- | --- | --- | --- | --- | --- |
| ESDIs-SOD | [Link](https://example.com/dataset-esdis-sod) | [PVTv2-B2](https://example.com/pvt-v2-b2) | 384x384 | channel=64, bs=32, lr=1.6e-4 | [Link](https://example.com/checkpoint-esdis-sod) | [Link](https://example.com/pred-esdis-sod) |
| CrackSeg9k | [Link](https://example.com/dataset-crackseg9k) | [PVTv2-B2](https://example.com/pvt-v2-b2) | 384x384 | channel=64, bs=32, lr=1.6e-4 | [Link](https://example.com/checkpoint-crackseg9k) | [Link](https://example.com/pred-crackseg9k) |
| ZJU-Leaper | [Link](https://example.com/dataset-zju-leaper) | [PVTv2-B2](https://example.com/pvt-v2-b2) | 384x384 | channel=64, bs=32, lr=1.6e-4 | [Link](https://example.com/checkpoint-zju-leaper) | [Link](https://example.com/pred-zju-leaper) |

## Citation

If this code is helpful, please cite:

```bibtex
@inproceedings{yan2025wavelet,
  title={Wavelet and Prototype Augmented Query-based Transformer for Pixel-level Surface Defect Detection},
  author={Yan, Feng and Jiang, Xiaoheng and Lu, Yang and Cao, Jiale and Chen, Dong and Xu, Mingliang},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={23860--23869},
  year={2025}
}
```

## Acknowledgement

This README follows the style of the official WPFormer repository.
