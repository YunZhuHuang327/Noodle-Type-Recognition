"""
ResNet-18 模型實作（從零開始訓練，不使用預訓練權重）
用於三種麵條分類：義大利麵(spaghetti)、拉麵(ramen)、烏龍麵(udon)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """ResNet的基本殘差塊"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # 第一層卷積
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 第二層卷積
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 下採樣層（用於匹配維度）
        self.downsample = downsample

    def forward(self, x):
        identity = x

        # 主路徑
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 捷徑連接
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    """ResNet-18 架構"""

    def __init__(self, num_classes=3):
        super(ResNet18, self).__init__()

        self.in_channels = 64

        # 初始卷積層
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet-18 的四個殘差層
        # layer1: [2 blocks] 64 channels
        self.layer1 = self._make_layer(64, 2, stride=1)
        # layer2: [2 blocks] 128 channels
        self.layer2 = self._make_layer(128, 2, stride=2)
        # layer3: [2 blocks] 256 channels
        self.layer3 = self._make_layer(256, 2, stride=2)
        # layer4: [2 blocks] 512 channels
        self.layer4 = self._make_layer(512, 2, stride=2)

        # 全局平均池化和分類層
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.3)  # 降低 Dropout 率防止欠擬合
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        # 權重初始化
        self._initialize_weights()

    def _make_layer(self, out_channels, blocks, stride=1):
        """構建殘差層"""
        downsample = None

        # 如果維度不匹配，需要下採樣
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers = []
        # 第一個block可能需要下採樣
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion

        # 後續blocks
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Kaiming初始化（適合ReLU）"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 初始特徵提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 四個殘差層
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局池化和分類
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # 在全連接層前應用Dropout
        x = self.fc(x)

        return x


def create_resnet18(num_classes=3):
    """創建ResNet-18模型"""
    model = ResNet18(num_classes=num_classes)
    return model


if __name__ == '__main__':
    # 測試模型
    model = create_resnet18(num_classes=3)
    print(f"模型參數量: {sum(p.numel() for p in model.parameters()):,}")

    # 測試前向傳播
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"輸入形狀: {dummy_input.shape}")
    print(f"輸出形狀: {output.shape}")
