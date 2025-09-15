import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# 读取数据
path_openfile_name = '150-HighFre.xlsx'
data = pd.read_excel(path_openfile_name)

# 数据预处理
X = data.iloc[:, 2:-3].values
print(X)
y = data.iloc[:, -3:].values
print(y)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 转换为PyTorch张量
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.3, random_state=42)

# 创建DataLoaders
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


# 定义一维卷积特征提取模型
class CNN1DFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNN1DFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * (X_train.shape[1] // 2), 100)

    def forward(self, x):
        x = x.unsqueeze(1)  # Adding a channel dimension
        x = self.pool(self.relu(self.conv1(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return x

# 实例化特征提取模型
feature_extractor = CNN1DFeatureExtractor()
criterion = nn.MSELoss()
optimizer = optim.Adam(feature_extractor.parameters(), lr=0.001)

def train_feature_extractor(num_epochs):
    for epoch in range(num_epochs):
        feature_extractor.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = feature_extractor(inputs)
            loss = criterion(outputs, outputs)  # 自编码器的损失，输入即输出
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# 训练特征提取模型
train_feature_extractor(50)

# 提取训练集和测试集的特征
def extract_features(loader):
    features = []
    targets = []
    feature_extractor.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = feature_extractor(inputs)
            features.append(outputs)
            targets.append(labels)
    return torch.cat(features), torch.cat(targets)

X_train_features, y_train_targets = extract_features(train_loader)
X_test_features, y_test_targets = extract_features(test_loader)

# 将提取的特征转换为numpy数组
X_train_np = X_train_features.numpy()
y_train_np = y_train_targets.numpy()
X_test_np = X_test_features.numpy()
y_test_np = y_test_targets.numpy()

# 使用XGBoost进行回归
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)

# 对每个输出特征单独训练一个XGBoost模型
xgb_models = []
for i in range(y_train_np.shape[1]):
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    model.fit(X_train_np, y_train_np[:, i])
    xgb_models.append(model)

# 预测
y_pred = np.column_stack([model.predict(X_test_np) for model in xgb_models])

# 评估模型
mse = mean_squared_error(y_test_np, y_pred, multioutput='raw_values')
print(f'Mean Squared Error for each output: {mse}')

# 可视化结果
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建一个窗口，包含三个子图
plt.figure(figsize=(15, 5))

# 第一个子图 - 归一化相速
plt.subplot(1, 3, 1)
plt.plot(y_test_np[:, 0].flatten(), 'o-', label='Real')
plt.plot(y_pred[:, 0].flatten(), '.-', label='Predicted')
plt.title("归一化相速 y_Real & y_Predict")
plt.legend()

# 第二个子图 - 耦合阻抗
plt.subplot(1, 3, 2)
plt.plot(y_test_np[:, 1].flatten(), 'o-', label='Real')
plt.plot(y_pred[:, 1].flatten(), '.-', label='Predicted')
plt.title("耦合阻抗 y_Real & y_Predict")
plt.legend()

# 第三个子图 - 衰减常数
plt.subplot(1, 3, 3)
plt.plot(y_test_np[:, 2].flatten(), 'o-', label='Real')
plt.plot(y_pred[:, 2].flatten(), '.-', label='Predicted')
plt.title("衰减常数 y_Real & y_Predict")
plt.legend()

plt.tight_layout()  # 调整子图之间的间距
plt.show()

# 保存特征提取模型状态字典
torch.save(feature_extractor.state_dict(), 'feature_extractor_state_dict.pth')
