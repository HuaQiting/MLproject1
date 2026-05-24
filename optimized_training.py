import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


class SEEDDataset(Dataset):
    """增强版SEED数据集"""
    def __init__(self, h5_path, augment=False):
        with h5py.File(h5_path, "r") as f:
            self.x = torch.tensor(f["X"][()], dtype=torch.float32)
            self.y = torch.tensor(f["y"][()], dtype=torch.long)
        
        self.mean = self.x.mean(dim=(0, 2), keepdim=True)
        self.std = self.x.std(dim=(0, 2), keepdim=True) + 1e-6
        self.x = (self.x - self.mean) / self.std
        self.augment = augment

    def __len__(self):
        return len(self.x)

    def _augment(self, x):
        if np.random.random() < 0.5:
            x = x + torch.randn_like(x) * 0.03
        
        if np.random.random() < 0.3:
            scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
            x = x * scale
        
        if np.random.random() < 0.25:
            time_shift = np.random.randint(-10, 10)
            if time_shift > 0:
                x = torch.roll(x, time_shift, dims=1)
                x[:, :time_shift] = 0
            elif time_shift < 0:
                x = torch.roll(x, time_shift, dims=1)
                x[:, time_shift:] = 0
        
        return x

    def __getitem__(self, idx):
        x = self.x[idx].clone()
        if self.augment:
            x = self._augment(x)
        return x, self.y[idx]


class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵"""
    def __init__(self, smoothing=0.15):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_probs = F.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


class ImprovedEEGNet(nn.Module):
    """改进的EEG分类模型"""
    def __init__(self, num_classes=3):
        super().__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv1d(62, 64, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.35)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.35)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=4, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


def train_with_cross_validation(n_folds=5, epochs=150, batch_size=32, lr=1e-3):
    """使用交叉验证的训练"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\nLoading SEED dataset...")
    full_dataset = SEEDDataset("SEED/train.h5", augment=False)
    
    print(f"Total samples: {len(full_dataset)}")
    print(f"Data shape: {full_dataset.x.shape}")
    
    # 5折交叉验证
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    all_preds = []
    all_labels = []
    
    print("\n" + "=" * 70)
    print(f"Starting {n_folds}-Fold Cross Validation")
    print("=" * 70)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(full_dataset.x, full_dataset.y)):
        print(f"\nFold {fold + 1}/{n_folds}")
        print("-" * 50)
        
        # 划分数据集
        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        test_subset = torch.utils.data.Subset(full_dataset, test_idx)
        
        train_subset.dataset.augment = True
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size)
        
        # 初始化模型
        model = ImprovedEEGNet(num_classes=3).to(device)
        
        # 标签平滑损失
        criterion = LabelSmoothingCrossEntropy(smoothing=0.15)
        
        # 优化器
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.005)
        
        # 学习率调度
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=30, T_mult=2, eta_min=1e-6
        )
        
        best_acc = 0.0
        best_model_state = None
        patience = 25
        no_improve = 0
        
        for epoch in range(epochs):
            model.train()
            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            scheduler.step()
            
            model.eval()
            correct = 0
            with torch.no_grad():
                for data, labels in test_loader:
                    data, labels = data.to(device), labels.to(device)
                    outputs = model(data)
                    correct += (outputs.argmax(1) == labels).sum().item()
            
            acc = correct / len(test_subset)
            
            if acc > best_acc:
                best_acc = acc
                best_model_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: Acc={acc:.4f}, Best={best_acc:.4f}")
            
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        fold_results.append(best_acc)
        
        # 使用最佳模型预测
        model.load_state_dict(best_model_state)
        model.eval()
        
        fold_preds = []
        fold_labels = []
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                fold_preds.extend(outputs.argmax(1).cpu().numpy())
                fold_labels.extend(labels.cpu().numpy())
        
        all_preds.extend(fold_preds)
        all_labels.extend(fold_labels)
        
        print(f"  Fold {fold + 1} Best Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    
    # 总结
    print("\n" + "=" * 70)
    print("Cross Validation Results:")
    print("=" * 70)
    
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    
    for i, acc in enumerate(fold_results):
        print(f"  Fold {i+1}: {acc:.4f} ({acc*100:.2f}%)")
    
    print(f"\n  Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  ({mean_acc*100:.2f}% ± {std_acc*100:.2f}%)")
    
    # 总体分类报告
    print("\n" + "=" * 70)
    print("Overall Classification Report (All Folds)")
    print("=" * 70)
    
    target_names = ['negative', 'neutral', 'positive']
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 交叉验证结果
    axes[0].bar([f'Fold {i+1}' for i in range(n_folds)], fold_results, color='steelblue', alpha=0.8)
    axes[0].axhline(mean_acc, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_acc:.2%}')
    axes[0].set_title(f'{n_folds}-Fold Cross Validation Results', fontweight='bold')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim([0, 1])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 混淆矩阵
    im = axes[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1].set_title('Confusion Matrix', fontweight='bold')
    tick_marks = np.arange(3)
    axes[1].set_xticks(tick_marks)
    axes[1].set_xticklabels(target_names, rotation=45, ha='right')
    axes[1].set_yticks(tick_marks)
    axes[1].set_yticklabels(target_names)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1].text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    plt.colorbar(im, ax=axes[1])
    
    # 类别准确率
    class_accs = cm.diagonal() / cm.sum(axis=1)
    axes[2].bar(target_names, class_accs, color=['red', 'gray', 'green'])
    axes[2].set_title('Per-Class Accuracy', fontweight='bold')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_ylim([0, 1])
    for i, v in enumerate(class_accs):
        axes[2].text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('cv_results.png', dpi=150, bbox_inches='tight')
    print(f"\nResults saved to 'cv_results.png'")
    plt.close()
    
    return mean_acc, std_acc, fold_results


if __name__ == "__main__":
    print("=" * 70)
    print("Improved Training Strategy with Cross-Validation")
    print("=" * 70)
    
    mean_acc, std_acc, fold_results = train_with_cross_validation(
        n_folds=5,
        epochs=150,
        batch_size=32,
        lr=1e-3
    )
    
    print("\n" + "=" * 70)
    print("Final Results:")
    print(f"  Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  ({mean_acc*100:.2f}% ± {std_acc*100:.2f}%)")
    print("=" * 70)
    print("\nDone!")
