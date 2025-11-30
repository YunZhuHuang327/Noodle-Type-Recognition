"""
模型評估腳本
計算混淆矩陣、分類報告等指標
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from resnet18_model import create_resnet18


def get_val_transform():
    """驗證集資料轉換"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def evaluate_model(model, data_loader, device='cuda'):
    """評估模型"""

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """繪製混淆矩陣"""

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('預測類別', fontsize=12)
    plt.ylabel('真實類別', fontsize=12)
    plt.title('混淆矩陣', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'混淆矩陣已儲存至: {save_path}')
    plt.close()


def main():
    # 設定
    DATA_DIR = r'd:\麵\dataset2025\train'
    MODEL_PATH = r'd:\麵\best_resnet18_noodles.pth'
    BATCH_SIZE = 32

    # 設定設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用設備: {device}')

    # 載入模型
    print('\n載入模型...')
    model = create_resnet18(num_classes=3)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"已載入模型（Epoch {checkpoint['epoch']}, 最佳驗證準確率: {checkpoint['best_acc']:.4f}）")

    # 載入完整訓練集（用於評估）
    print('\n載入資料集...')
    transform = get_val_transform()
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4)

    class_names = ['spaghetti', 'ramen', 'udon']
    print(f'類別: {dataset.classes}')
    print(f'資料集大小: {len(dataset)}')

    # 評估
    print('\n評估模型...')
    predictions, labels = evaluate_model(model, data_loader, device)

    # 計算準確率
    accuracy = np.mean(predictions == labels)
    print(f'\n整體準確率: {accuracy:.4f} ({accuracy*100:.2f}%)')

    # 分類報告
    print('\n分類報告:')
    print(classification_report(labels, predictions,
                               target_names=class_names, digits=4))

    # 繪製混淆矩陣
    plot_confusion_matrix(labels, predictions, class_names)

    # 每個類別的準確率
    print('\n各類別準確率:')
    for i, class_name in enumerate(class_names):
        class_mask = labels == i
        class_acc = np.mean(predictions[class_mask] == labels[class_mask])
        print(f'{class_name}: {class_acc:.4f} ({class_acc*100:.2f}%)')


if __name__ == '__main__':
    main()
