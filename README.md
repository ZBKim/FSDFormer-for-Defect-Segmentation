# WPFormer Defect Segmentation

本项目使用 WPFormer 进行缺陷/显著目标分割训练与验证，当前训练入口为 `defect_train_multiGPU.py`，推理入口为 `defect_test.py`。

## 1. 环境准备

建议使用 Conda 创建独立环境：

```bash
conda create -n wpformer python=3.8 -y
conda activate wpformer
```

安装依赖：

```bash
pip install torch torchvision torchaudio
pip install timm opencv-python pillow numpy tqdm tensorboard albumentations scipy
```

如果服务器 CUDA/PyTorch 版本有固定要求，请按本机 CUDA 版本安装对应的 PyTorch。

## 2. 项目目录

在项目根目录下创建预训练模型和数据集目录：

```bash
mkdir -p pretrainModel datasets
```

当前代码实际使用的数据集目录名区分大小写，推荐目录结构如下：

```text
WPFormer1
├── pretrainModel
│   ├── pvt_v2_b2.pth
│   └── pvt_v2_b4.pth
├── datasets
│   ├── CrackSeg9k
│   │   ├── train
│   │   │   ├── images
│   │   │   └── gt
│   │   └── test
│   │       ├── images
│   │       └── gt
│   ├── ESDIs-SOD
│   │   ├── train
│   │   │   ├── images
│   │   │   └── gt
│   │   └── test
│   │       ├── images
│   │       └── gt
│   └── ZJU-Leaper
│       ├── train
│       │   ├── images
│       │   └── gt
│       └── test
│           ├── images
│           └── gt
```

注意：训练脚本读取的是每个数据集下的 `train/images`、`train/gt`、`test/images`、`test/gt`。图片和标签文件需要一一对应，并且排序后数量一致。

## 3. 放置 PVTv2 预训练模型

将 PVTv2 预训练权重上传到 `pretrainModel`：

```text
pretrainModel/pvt_v2_b2.pth
pretrainModel/pvt_v2_b4.pth
```

当前训练脚本默认创建：

```python
WPFormer(method="pvt_v2_b2", channel=64)
```

因此至少需要 `pvt_v2_b2.pth`。如果切换到 `pvt_v2_b4`，则需要 `pvt_v2_b4.pth`。

## 4. 修改代码中的路径

当前仓库路径为 `/raid/jzb4/WPFormer1`。代码里仍有部分硬编码路径指向 `/raid/jzb4/WPFormer`，训练和验证前需要同步修改。

### 4.1 修改预训练模型路径

打开 `model/WPFormer6.py`，将：

```python
path = '/raid/jzb4/WPFormer/pretrainModel/pvt_v2_b2.pth'
path = '/raid/jzb4/WPFormer/pretrainModel/pvt_v2_b4.pth'
```

改为：

```python
path = './pretrainModel/pvt_v2_b2.pth'
path = './pretrainModel/pvt_v2_b4.pth'
```

如果使用 `model/WPFormer6_before_arch_enhance.py` 或 `model/WPFormer6_arch_v1_aspp_boundary_query.py`，也需要做同样修改。

### 4.2 修改训练数据集路径

打开 `defect_train_multiGPU.py`，将：

```python
root = "/raid/jzb4/WPFormer/datasets"
```

改为：

```python
root = "./datasets"
```

## 5. 修改训练参数

训练参数位于 `defect_train_multiGPU.py` 文件末尾：

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
dataset = "ESDIs-SOD"
modelname = "WPFormer6"
message = "WPFormer"
bs = 32
lr = 1.6e-4
```

常用修改项：

```python
# 选择 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# 选择数据集，必须和 datasets 下目录名一致
dataset = "CrackSeg9k"
# dataset = "ESDIs-SOD"
# dataset = "ZJU-Leaper"

# 单卡基础 batch size；脚本会按 GPU 数自动放大
bs = 32

# 基础学习率；脚本会按 sqrt(GPU 数) 自动缩放
lr = 1.6e-4

# 每个数据集的总 epoch 和开始验证 epoch
epoch_cfg = {
    "CrackSeg9k": (250, 100),
    "ZJU-Leaper": (250, 100),
    "ESDIs-SOD": (350, 150)
}
```

脚本中的实际 batch size 和学习率计算方式为：

```python
actual_batch = base_batch * max(1, gpu_cnt)
actual_lr = base_lr * (gpu_cnt ** 0.5)
```

如果显存不足，优先降低 `bs`。

## 6. 开始训练

在项目根目录执行：

```bash
python defect_train_multiGPU.py
```

训练输出默认保存在：

```text
ablationruns_esdis/{运行时间}_{实验名}/
├── train.log
├── val_metrics.csv
├── events.out.tfevents...
└── checkpoints
    └── {S值}_{model_name}_best.pth
```

其中 `checkpoints` 下保存的是验证集 S-measure 最优的模型权重。

可使用 TensorBoard 查看训练过程：

```bash
tensorboard --logdir ablationruns_esdis
```

## 7. 训练中验证

`defect_train_multiGPU.py` 会在训练过程中自动对对应数据集的 `test` 集进行验证，计算：

- `MAE`
- `S-measure`

验证结果会写入：

```text
ablationruns_esdis/{运行时间}_{实验名}/val_metrics.csv
```

日志会写入：

```text
ablationruns_esdis/{运行时间}_{实验名}/train.log
```

## 8. 单独推理/生成预测图

如果需要使用训练好的权重单独推理，先检查 `defect_test.py`。

当前文件中需要修改三处：

1. 模型导入路径：

```python
from model.WPFormer6 import WPFormer
```

2. 数据集路径：

```python
file_dir = "./datasets"
```

3. 预测结果保存路径：

```python
save_dir = os.path.join("./ablationPred", model_name, dataset_name)
```

然后修改待加载权重：

```python
model_save = "./ablationruns_esdis/你的实验目录/checkpoints/你的_best.pth"
```

运行：

```bash
python defect_test.py
```

预测图会保存到：

```text
ablationPred/{model_name}/{dataset_name}/
```

## 9. 单独计算指标

`eval.py` 中提供了基于预测图和 GT 的指标计算函数 `test(test_pred_list, test_gt_list)`，可计算：

- `MAE`
- `wFmeasure`
- `Smeasure`
- `Fmeasure`
- `Emeasure`

使用前需要将 `eval.py` 末尾的 `pred_dir`、`test_gt_dir`、`datasetnames`、`methods` 改成自己的预测图和标签路径。

## 10. 常见问题

### FileNotFoundError: 找不到预训练模型

检查 `pretrainModel` 下是否存在：

```text
pvt_v2_b2.pth
```

并确认 `model/WPFormer6.py` 中的预训练模型路径已经改为当前项目路径。

### 找不到数据集目录

检查 `defect_train_multiGPU.py` 中：

```python
root = "./datasets"
```

并确认数据集目录名和代码中的 `dataset` 完全一致，例如：

```python
dataset = "ESDIs-SOD"
```

对应目录必须是：

```text
datasets/ESDIs-SOD
```

### Image/Gt 数量不一致

检查：

```text
train/images
train/gt
test/images
test/gt
```

图片与标签数量必须一致，且排序后能一一对应。建议图片和标签使用相同文件名，仅扩展名可不同。

### CUDA out of memory

降低 `defect_train_multiGPU.py` 中的：

```python
bs = 32
```

例如改为：

```python
bs = 16
```

### 多卡训练没有生效

确认：

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
```

并检查运行时日志中的：

```text
GPU=2
```

当前代码使用 `torch.nn.DataParallel`，可直接通过 `CUDA_VISIBLE_DEVICES` 控制可见 GPU。

## 11. 快速流程

```bash
# 1. 创建目录
mkdir -p pretrainModel datasets

# 2. 上传 PVTv2 权重
# pretrainModel/pvt_v2_b2.pth

# 3. 上传数据集
# datasets/ESDIs-SOD/train/images
# datasets/ESDIs-SOD/train/gt
# datasets/ESDIs-SOD/test/images
# datasets/ESDIs-SOD/test/gt

# 4. 修改 defect_train_multiGPU.py 中的 root、GPU、dataset、bs、lr

# 5. 训练
python defect_train_multiGPU.py

# 6. 查看日志和最优权重
ls ablationruns_esdis
```
