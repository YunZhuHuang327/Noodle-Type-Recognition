"""
ResNet-18 麵條分類訓練腳本
目標：在測試集上達到 95% 以上的準確率
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from resnet18_model import create_resnet18


# 類別名稱
CLASS_NAMES = ['0_spaghetti', '1_ramen', '2_udon']


def mixup_data(x, y, alpha=0.2):
    """
    MixUp 資料增強
    將兩張圖片及其標籤混合，提升模型泛化能力
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp 損失函數"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_data_transforms():
    """定義資料增強和預處理"""

    # 訓練集：強化的資料增強策略
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 增加裁剪多樣性
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),  # 增加旋轉角度
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 增加色彩變化
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 添加平移增強
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3))  # 降低擦除機率
    ])

    # 驗證集：僅調整大小和標準化
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def load_data(data_dir, batch_size=32, val_split=0.10):  # 降至 0.10 增加訓練資料
    """載入並分割資料集"""

    train_transform, val_transform = get_data_transforms()

    # 載入完整訓練集
    full_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transform
    )

    # 分割訓練集和驗證集
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 為驗證集設置正確的transform
    val_dataset.dataset = copy.deepcopy(full_dataset)
    val_dataset.dataset.transform = val_transform

    # 創建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"訓練集大小: {train_size}")
    print(f"驗證集大小: {val_size}")
    print(f"類別: {full_dataset.classes}")

    return train_loader, val_loader, full_dataset.classes


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=100, device='cuda', save_path='best_model.pth', use_mixup=True):
    """訓練模型"""

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # Early stopping參數
    patience = 30  # 減少patience，更早停止
    patience_counter = 0
    best_loss = float('inf')
    min_delta = 0.001  # 最小改善閾值

    print(f'MixUp: {"啟用" if use_mixup else "關閉"}')

    for epoch in range(num_epochs):
        epoch_start = time.time()

        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 60)

        # 訓練階段
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # 使用 MixUp 增強
            if use_mixup:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.2)
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                # 對於準確率計算，使用原始標籤
                _, preds = torch.max(outputs, 1)
                running_corrects += (lam * (preds == labels_a).sum().float() +
                                    (1 - lam) * (preds == labels_b).sum().float())
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 驗證階段
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())

        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        # 學習率調整
        scheduler.step()  # CosineAnnealingLR 不需要參數
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Learning Rate: {current_lr:.6f}')

        # 儲存最佳模型（基於驗證損失和準確率）
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'history': history
            }, save_path)
            print(f'*** 儲存最佳模型! 驗證準確率: {best_acc:.4f} ***')

        epoch_time = time.time() - epoch_start
        print(f'Epoch 時間: {epoch_time:.2f}s')

        # Early stopping
        if patience_counter >= patience:
            print(f'\n{patience} epochs內無改善，提前停止訓練')
            break

    print(f'\n訓練完成! 最佳驗證準確率: {best_acc:.4f}')

    # 載入最佳模型權重
    model.load_state_dict(best_model_wts)

    return model, history


def plot_history(history, save_path='training_history.png'):
    """繪製訓練歷史"""

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 損失曲線
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # 準確率曲線
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'訓練歷史圖表已儲存至: {save_path}')
    plt.close()


def main():
    # 設定 - 改善過擬合的參數
    DATA_DIR = r'd:\麵\dataset2025'
    BATCH_SIZE = 64  # 增加batch size有助於穩定訓練
    NUM_EPOCHS = 300  # 減少總epoch數
    LEARNING_RATE = 0.01  # 降低初始學習率
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4  # 增加權重衰減（L2正則化）

    # 設定設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用設備: {device}')
    print(f'\n=== 訓練參數 ===')
    print(f'Batch Size: {BATCH_SIZE}')
    print(f'Learning Rate: {LEARNING_RATE}')
    print(f'Weight Decay: {WEIGHT_DECAY}')
    print(f'Max Epochs: {NUM_EPOCHS}')

    # 載入資料
    print('\n載入資料集...')
    train_loader, val_loader, class_names = load_data(DATA_DIR, BATCH_SIZE)

    # 創建模型
    print('\n創建ResNet-18模型...')
    model = create_resnet18(num_classes=3)
    model = model.to(device)
    print(f'模型參數量: {sum(p.numel() for p in model.parameters()):,}')

    # 損失函數和優化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 添加標籤平滑
    # 使用 AdamW 優化器（比 SGD 更穩定且收斂更快）
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    # 如果想使用 SGD，請註解上面改用：
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
    #                      momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # 學習率調度器：使用 Cosine Annealing 實現平滑的學習率衰減
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )
    # 如果想使用 ReduceLROnPlateau，請註解上面改用：
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-6
    # )

    # 訓練模型
    print('\n開始訓練...')
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=NUM_EPOCHS, device=device, save_path='best_resnet18_noodles.pth'
    )

    # 繪製訓練歷史
    plot_history(history)

    print('\n訓練完成!')


if __name__ == '__main__':
    main()
