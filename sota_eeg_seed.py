import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import argparse
import os
from pathlib import Path

class SEEDDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(self.h5_path, "r") as f:
            self.x = torch.tensor(f["X"][()], dtype=torch.float32)
            self.y = torch.tensor(f["y"][()], dtype=torch.long)
        assert len(self.x) == len(self.y), "X and y length mismatch"

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class TestDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(self.h5_path, "r") as f:
            self.x = torch.tensor(f["X"][()], dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        attn_scores = self.attention(x)
        attn_scores = attn_scores.squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        output = torch.matmul(attn_weights.unsqueeze(1), x)
        return output.squeeze(1)

class SOTAEEGClassifier(nn.Module):
    def __init__(self, chans=62, time_points=400, num_classes=3,
                 cnn_channels=64, lstm_hidden_dim=128, lstm_num_layers=2,
                 dropout=0.5):
        super(SOTAEEGClassifier, self).__init__()

        self.time_conv = nn.Sequential(
            nn.Conv1d(chans, cnn_channels, kernel_size=64, padding=32, bias=False),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )

        self.residual_blocks = nn.Sequential(
            self._make_residual_block(cnn_channels, cnn_channels),
            self._make_residual_block(cnn_channels, cnn_channels * 2),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )

        self.se_block = self._make_se_block(cnn_channels * 2)

        self.lstm = nn.LSTM(
            input_size=cnn_channels * 2,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=True
        )

        self.temporal_attn = TemporalAttention(lstm_hidden_dim * 2)

        lstm_out_dim = lstm_hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        self._init_weights()

    def _make_residual_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_se_block(self, channels, reduction=16):
        return nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for name, param in self.lstm.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        x = self.time_conv(x)

        x = self.residual_blocks(x)

        se_weights = self.se_block(x).unsqueeze(-1)
        x = x * se_weights

        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)

        feat = self.temporal_attn(lstm_out)

        logits = self.classifier(feat)
        return logits

def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for data, label in train_loader:
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, label)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * data.size(0)
        preds = torch.argmax(output, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    avg_loss = total_loss / len(train_loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, acc, f1

def val_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, label in val_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)

            total_loss += loss.item() * data.size(0)
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    avg_loss = total_loss / len(val_loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, acc, f1

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser(description='SOTA EEG Classifier for SEED Dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio (10%-20%)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_model', action='store_true', help='Save the best model')
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    DATA_NAME = "SEED"
    DATA_DIR = Path(DATA_NAME)
    train_path = DATA_DIR / "train.h5"
    test_path = DATA_DIR / "test_x_only.h5"

    full_train_dataset = SEEDDataset(train_path)

    val_size = int(len(full_train_dataset) * args.val_ratio)
    train_size = len(full_train_dataset) - val_size

    print(f"\n{'='*50}")
    print(f"Dataset split:")
    print(f"Total training samples: {len(full_train_dataset)}")
    print(f"Train set: {train_size} samples ({(1-args.val_ratio)*100:.0f}%)")
    print(f"Validation set: {val_size} samples ({args.val_ratio*100:.0f}%)")
    print(f"{'='*50}")

    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = SOTAEEGClassifier(
        chans=62,
        time_points=400,
        num_classes=3,
        cnn_channels=64,
        lstm_hidden_dim=128,
        lstm_num_layers=2,
        dropout=0.5
    ).to(device)

    print(f"\nModel architecture:")
    print(f"- CNN: 1D temporal convolution + residual blocks")
    print(f"- SE Attention: Channel-wise attention")
    print(f"- RNN: BiLSTM + temporal attention")
    print(f"- Classifier: 3-layer MLP")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"- Total parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_val_acc = 0.0
    best_epoch = 0
    patience = 0
    max_patience = 20

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"\n{'='*50}")
    print("Training started...")
    print(f"{'='*50}")

    for epoch in range(args.epochs):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        val_loss, val_acc, val_f1 = val_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:02d}/{args.epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience = 0
            if args.save_model:
                torch.save(model.state_dict(), 'sota_eeg_best.pth')
        else:
            patience += 1

        if patience >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"{'='*50}")

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc', color='blue')
    plt.plot(val_accs, label='Val Acc', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sota_training_curve.png', dpi=150)

    print("\nGenerating classification report on validation set...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, label in val_loader:
            data = data.to(device)
            output = model(data)
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['negative', 'neutral', 'positive']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    print("\nTest set inference...")
    test_dataset = TestDataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()
    all_test_preds = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            pred = torch.argmax(output, dim=1)
            all_test_preds.extend(pred.cpu().numpy())

    output_path = DATA_DIR / 'SOTA_seed_predictions.txt'
    with open(output_path, 'w') as f:
        for pred in all_test_preds:
            f.write(f"{pred}\n")

    print(f"\nSaved {len(all_test_preds)} test predictions to {output_path}")
    print("Done!")

if __name__ == '__main__':
    main()