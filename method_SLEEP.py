import torch
import torch.nn as nn
import h5py
import numpy as np
from torch.utils.data import Dataset


# ==================== 数据集类 ====================
class TrainDataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            self.x = torch.tensor(f['X'][()], dtype=torch.float32)
            self.y = torch.tensor(f['y'][()], dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class TestDataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            self.x = torch.tensor(f['X'][()], dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


# ==================== EEGNet 模型 ====================
class EEGNet(nn.Module):
    """EEGNet 精简版 - 适合EEG信号特征提取"""

    def __init__(self, chans=6, time_points=1000, num_classes=5):
        super().__init__()
        # 时间卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16), nn.ELU(), nn.AvgPool2d((1, 2))
        )
        # 空间卷积 (深度可分离)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (chans, 1), groups=16, bias=False),
            nn.BatchNorm2d(32), nn.ELU(), nn.AvgPool2d((1, 2)), nn.Dropout(0.3)
        )
        # 可分离卷积
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 16), groups=32, padding=(0, 8), bias=False),
            nn.Conv2d(32, 32, 1, bias=False),
            nn.BatchNorm2d(32), nn.ELU(), nn.AdaptiveAvgPool2d((1, 8)), nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B,C,T) -> (B,1,C,T)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.classifier(x.flatten(1))


# ==================== 工具函数 ====================
def downsample(data, target_len=1000):
    """降采样到目标长度"""
    step = data.shape[-1] // target_len
    return data[:, :, ::step]


def load_dataset_info(data_dir):
    """加载数据集信息"""
    import json
    from pathlib import Path

    data_dir = Path(data_dir)
    info_path = data_dir / "dataset_info.json"

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    return {
        "category_list": info["dataset"]["category_list"],
        "channels": info["dataset"]["channels"],
        "target_sampling_rate": info["processing"]["target_sampling_rate"],
        "window_sec": info["processing"]["window_sec"],
    }


def get_class_weights(dataset):
    """计算类别权重用于平衡损失函数"""
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    y_train = dataset.y.numpy()
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return torch.tensor(weights, dtype=torch.float32)