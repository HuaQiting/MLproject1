import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold


def log_results(msg):
    print(msg)
    with open("training_results.txt", "a") as f:
        f.write(msg + "\n")


def train_and_evaluate():
    with open("training_results.txt", "w") as f:
        f.write("")
    
    log_results("=" * 60)
    log_results("ENHANCED EEGNET - SEED Dataset")
    log_results("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_results(f"Device: {device}")
    
    try:
        log_results("\nLoading dataset...")
        data_path = "C:/Users/25447/Desktop/学习/机器学习及医学工程应用/共选题/MLproject1/SEED/train.h5"
        log_results(f"Data path: {data_path}")
        log_results(f"Path exists: {os.path.exists(data_path)}")
        
        with h5py.File(data_path, "r") as f:
            X = f["X"][()]
            y = f["y"][()]
        log_results(f"Data loaded successfully!")
        log_results(f"X shape: {X.shape}")
        log_results(f"y shape: {y.shape}")
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        mean = X_tensor.mean(dim=(0, 2), keepdim=True)
        std = X_tensor.std(dim=(0, 2), keepdim=True) + 1e-6
        X_tensor = (X_tensor - mean) / std
        
        class SEEDDataset(Dataset):
            def __init__(self, x, y, augment=False):
                self.x = x
                self.y = y
                self.augment = augment
            
            def __len__(self):
                return len(self.x)
            
            def _augment(self, x):
                if np.random.random() < 0.5:
                    x = x + torch.randn_like(x) * 0.03
                if np.random.random() < 0.3:
                    scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
                    x = x * scale
                return x
            
            def __getitem__(self, idx):
                x = self.x[idx].clone()
                if self.augment:
                    x = self._augment(x)
                return x, self.y[idx]
        
        dataset = SEEDDataset(X_tensor, y_tensor)
        log_results(f"Dataset ready: {len(dataset)} samples")
        
        n_folds = 5
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_results = []
        best_model_state = None
        best_fold_idx = 0
        best_fold_acc = 0.0
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(dataset.x, dataset.y)):
            log_results(f"\nFOLD {fold + 1}/{n_folds}")
            log_results("-" * 40)
            
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            test_subset = torch.utils.data.Subset(dataset, test_idx)
            train_subset.dataset.augment = True
            
            train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_subset, batch_size=32)
            
            class EnhancedEEGNet(nn.Module):
                def __init__(self, num_classes=3):
                    super().__init__()
                    self.init_conv = nn.Sequential(
                        nn.Conv1d(62, 48, kernel_size=11, padding=5),
                        nn.BatchNorm1d(48),
                        nn.ELU(),
                        nn.MaxPool1d(2),
                        nn.Dropout(0.25)
                    )
                    self.res1 = nn.Sequential(
                        nn.Conv1d(48, 96, kernel_size=5, padding=2, stride=2),
                        nn.BatchNorm1d(96),
                        nn.ELU()
                    )
                    self.res2 = nn.Sequential(
                        nn.Conv1d(96, 96, kernel_size=5, padding=2),
                        nn.BatchNorm1d(96),
                        nn.ELU()
                    )
                    self.res3 = nn.Sequential(
                        nn.Conv1d(96, 192, kernel_size=3, padding=1, stride=2),
                        nn.BatchNorm1d(192),
                        nn.ELU()
                    )
                    self.classifier = nn.Sequential(
                        nn.AdaptiveAvgPool1d(1),
                        nn.Flatten(),
                        nn.Linear(192, 96),
                        nn.ELU(),
                        nn.Dropout(0.35),
                        nn.Linear(96, 3)
                    )
                
                def forward(self, x):
                    x = self.init_conv(x)
                    x = self.res1(x)
                    x = self.res2(x)
                    x = self.res3(x)
                    x = self.classifier(x)
                    return x
            
            model = EnhancedEEGNet().to(device)
            total_params = sum(p.numel() for p in model.parameters())
            log_results(f"Model created: {total_params:,} parameters")
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.005)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15)
            
            best_acc = 0.0
            best_state = None
            patience = 30
            no_improve = 0
            
            for epoch in range(150):
                model.train()
                for data, labels in train_loader:
                    data, labels = data.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                
                model.eval()
                correct = 0
                with torch.no_grad():
                    for data, labels in test_loader:
                        data, labels = data.to(device), labels.to(device)
                        outputs = model(data)
                        correct += (outputs.argmax(1) == labels).sum().item()
                
                acc = correct / len(test_loader.dataset)
                
                if acc > best_acc:
                    best_acc = acc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                
                if (epoch + 1) % 20 == 0:
                    log_results(f"  Epoch {epoch+1}: Acc={acc:.4f}, Best={best_acc:.4f}")
                
                if no_improve >= patience:
                    log_results(f"  Early stopping at epoch {epoch+1}")
                    break
            
            fold_results.append(best_acc)
            
            if best_acc > best_fold_acc:
                best_fold_idx = fold
                best_fold_acc = best_acc
                best_model_state = best_state
            
            log_results(f"  Fold {fold+1} Best: {best_acc:.4f} ({best_acc*100:.2f}%)")
        
        log_results("\n" + "=" * 60)
        log_results("FINAL RESULTS")
        log_results("=" * 60)
        
        mean_acc = np.mean(fold_results)
        std_acc = np.std(fold_results)
        
        for i, acc in enumerate(fold_results):
            log_results(f"  Fold {i+1}: {acc:.4f} ({acc*100:.2f}%)")
        
        log_results(f"\n  MEAN ACCURACY: {mean_acc:.4f} ± {std_acc:.4f}")
        log_results(f"  ({mean_acc*100:.2f}% ± {std_acc*100:.2f}%)")
        
        # -----------------------------------------
        # 加载并预测 test_x_only.h5
        # -----------------------------------------
        log_results("\n" + "=" * 60)
        log_results("Predicting on test_x_only.h5")
        log_results("=" * 60)
        
        log_results(f"Using best model from Fold {best_fold_idx + 1} with accuracy: {best_fold_acc:.4f}")
        
        # 加载最佳模型
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        model.eval()
        
        output_path = "C:/Users/25447/Desktop/学习/机器学习及医学工程应用/共选题/MLproject1/optimized_predictions.txt"
        
        # 加载测试数据
        test_data_path = "C:/Users/25447/Desktop/学习/机器学习及医学工程应用/共选题/MLproject1/SEED/test_x_only.h5"
        log_results(f"\nLoading test data from: {test_data_path}")
        
        with h5py.File(test_data_path, "r") as f:
            X_test = torch.tensor(f['X'][:], dtype=torch.float32)
        
        log_results(f"Test data shape: {X_test.shape}")
        
        # 使用相同的标准化（基于训练数据）
        X_test = (X_test - mean) / std
        
        # 创建测试数据集（只有X）
        class TestDataset(Dataset):
            def __init__(self, x):
                self.x = x
            
            def __len__(self):
                return len(self.x)
            
            def __getitem__(self, idx):
                return self.x[idx]
        
        test_dataset = TestDataset(X_test)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        # 预测测试数据
        test_preds = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                outputs = model(data)
                preds = torch.argmax(outputs, dim=1)
                test_preds.extend(preds.cpu().tolist())
        
        # 保存测试预测结果
        with open(output_path, "w", encoding="utf-8") as f:
            for label in test_preds:
                f.write(f"{int(label)}\n")
        
        log_results(f"Saved {len(test_preds)} test predictions to: {output_path}")
        
        return mean_acc, std_acc
    
    except Exception as e:
        log_results(f"\nERROR: {str(e)}")
        import traceback
        log_results(traceback.format_exc())
        return None, None


if __name__ == "__main__":
    train_and_evaluate()