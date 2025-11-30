"""
ResNet-18 測試集預測腳本
對測試集進行預測並生成提交檔案
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from resnet18_model import create_resnet18
from tqdm import tqdm


class TestDataset(Dataset):
    """測試集資料集"""

    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.test_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_name


def get_test_transform():
    """測試集資料轉換（與訓練時驗證集相同）"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def get_tta_transforms():
    """測試時增強（TTA）- 多種轉換方式"""
    return [
        # 1. 標準中心裁剪
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ]),
        # 2. 較大尺寸
        transforms.Compose([
            transforms.Resize(288),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ]),
        # 3. 水平翻轉
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ]),
    ]


def predict_test_set(model, test_loader, device='cuda'):
    """對測試集進行預測"""

    model.eval()
    predictions = []
    filenames = []

    with torch.no_grad():
        for images, img_names in tqdm(test_loader, desc='預測中'):
            images = images.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            predictions.extend(preds.cpu().numpy())
            filenames.extend(img_names)

    return filenames, predictions


def predict_test_set_with_tta(model, test_dir, device='cuda', batch_size=64):
    """使用測試時增強（TTA）進行預測 - 提升準確率"""

    model.eval()
    tta_transforms = get_tta_transforms()

    # 儲存所有 TTA 的結果
    all_predictions = []
    image_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])

    print(f'\n使用 {len(tta_transforms)} 種測試增強方式...')

    for idx, transform in enumerate(tta_transforms):
        print(f'\n--- TTA {idx+1}/{len(tta_transforms)} ---')

        # 為每種轉換創建資料集
        test_dataset = TestDataset(test_dir, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=4)

        predictions = []
        with torch.no_grad():
            for images, _ in tqdm(test_loader, desc=f'TTA {idx+1}'):
                images = images.to(device)
                outputs = model(images)
                predictions.append(outputs.cpu())

        # 合併所有批次的預測
        predictions = torch.cat(predictions, dim=0)
        all_predictions.append(predictions)

    # 平均所有 TTA 的預測結果
    print('\n合併 TTA 預測結果...')
    avg_predictions = torch.stack(all_predictions).mean(dim=0)
    final_preds = torch.argmax(avg_predictions, dim=1).numpy()

    return image_files, final_preds


def create_submission(filenames, predictions, output_file='submission.csv'):
    """創建提交檔案"""

    # 從檔名中提取ID (test_0000.jpg -> 0)
    ids = [int(fname.replace('test_', '').replace('.jpg', '')) for fname in filenames]

    # 創建DataFrame，格式: ID,Target
    df = pd.DataFrame({
        'ID': ids,
        'Target': predictions
    })

    # 按ID排序
    df = df.sort_values('ID')

    # 儲存（不含index）
    df.to_csv(output_file, index=False)
    print(f'\n提交檔案已儲存至: {output_file}')

    # 顯示統計
    print('\n預測統計:')
    class_mapping = {0: 'spaghetti', 1: 'ramen', 2: 'udon'}
    print('各類別預測數量:')
    for class_id, count in df['Target'].value_counts().sort_index().items():
        print(f'  {class_id} ({class_mapping[class_id]}): {count}')

    return df


def main():
    # 設定
    TEST_DIR = r'd:\麵\dataset2025\test\unknown'
    MODEL_PATH = r'd:\麵\best_resnet18_noodles.pth'
    BATCH_SIZE = 64
    OUTPUT_FILE = r'd:\麵\submission.csv'
    USE_TTA = True  # 設定為 True 啟用測試時增強，False 使用標準預測

    # 設定設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用設備: {device}')
    print(f'TTA 模式: {"啟用" if USE_TTA else "關閉"}')

    # 載入模型
    print('\n載入模型...')
    model = create_resnet18(num_classes=3)

    # 載入訓練好的權重
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"已載入模型（Epoch {checkpoint['epoch']}, 最佳驗證準確率: {checkpoint['best_acc']:.4f}）")

    # 預測
    if USE_TTA:
        print('\n開始預測（使用 TTA）...')
        filenames, predictions = predict_test_set_with_tta(
            model, TEST_DIR, device=device, batch_size=BATCH_SIZE
        )
    else:
        # 準備測試資料
        print('\n準備測試資料...')
        test_transform = get_test_transform()
        test_dataset = TestDataset(TEST_DIR, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=4)

        print(f'測試集大小: {len(test_dataset)}')

        print('\n開始預測（標準模式）...')
        filenames, predictions = predict_test_set(model, test_loader, device)

    # 創建提交檔案
    df = create_submission(filenames, predictions, OUTPUT_FILE)

    # 顯示前幾筆預測
    print('\n前10筆預測:')
    print(df.head(10))

    print('\n完成!')


if __name__ == '__main__':
    main()
