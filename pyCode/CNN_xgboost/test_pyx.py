import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

def read_txt_file(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as file:
        lines = file.readlines()
    # 第一行为表头
    header = lines[0].strip().split()
    # 跳过第一行
    data_lines = lines[1:]
    data = []
    for line in data_lines:
        numbers = [float(num) for num in line.split() if num]
        data.append(numbers)
    return header, np.array(data)

def train_feature_extractor(feature_extractor, train_loader, criterion, optimizer, num_epochs=50):
    total_loss = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        feature_extractor.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = feature_extractor(inputs)
            loss = criterion(outputs, outputs)  # 自编码器的损失，输入即输出
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        total_loss.append(epoch_loss / len(train_loader))
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
    return total_loss

def extract_features(loader, feature_extractor):
    features = []
    targets = []
    feature_extractor.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = feature_extractor(inputs)
            features.append(outputs)
            targets.append(labels)
    return torch.cat(features), torch.cat(targets)

class CNN1DFeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super(CNN1DFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * (input_dim // 2), 100)

    def forward(self, x):
        x = x.unsqueeze(1)  # Adding a channel dimension
        x = self.pool(self.relu(self.conv1(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return x

def save_models(feature_extractor, xgb_models, file_path):
    models_dict = {
        'feature_extractor_state_dict': feature_extractor.state_dict(),
        'xgb_models': [model.get_booster().save_raw() for model in xgb_models]
    }
    torch.save(models_dict, file_path)

def load_models(feature_extractor, file_path):
    models_dict = torch.load(file_path)
    feature_extractor.load_state_dict(models_dict['feature_extractor_state_dict'])
    xgb_models = []
    for model_str in models_dict['xgb_models']:
        model = xgb.XGBRegressor(objective='reg:squarederror')
        model.load_model(model_str)
        xgb_models.append(model)
    return feature_extractor, xgb_models

def train_and_predict(input_data_path, output_data_path, model_save_path='models.pth'):
    input_header, input_data = read_txt_file(input_data_path)
    output_header, output_data = read_txt_file(output_data_path)

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(input_data)

    # 转换为PyTorch张量
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(output_data, dtype=torch.float32)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.3, random_state=42)

    # 创建DataLoaders
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # 实例化特征提取模型
    feature_extractor = CNN1DFeatureExtractor(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(feature_extractor.parameters(), lr=0.001)

    # 训练特征提取模型
    total_loss = train_feature_extractor(feature_extractor, train_loader, criterion, optimizer, num_epochs=50)

    # 提取训练集和测试集的特征
    X_train_features, y_train_targets = extract_features(train_loader, feature_extractor)
    X_test_features, y_test_targets = extract_features(test_loader, feature_extractor)

    # 将提取的特征转换为numpy数组
    X_train_np = X_train_features.numpy()
    y_train_np = y_train_targets.numpy()
    X_test_np = X_test_features.numpy()
    y_test_np = y_test_targets.numpy()

    # 使用XGBoost进行回归
    xgb_models = []
    for i in range(y_train_np.shape[1]):
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
        model.fit(X_train_np, y_train_np[:, i])
        xgb_models.append(model)

    save_models(feature_extractor, xgb_models, model_save_path)
    print(f'Models saved to {model_save_path}')

    # 预测
    y_pred = np.column_stack([model.predict(X_test_np) for model in xgb_models])

    # 评估模型
    mse = mean_squared_error(y_test_np, y_pred)
    # 计算每列数据的单独 MSE
    mse_columns = [mean_squared_error(y_test_np[:, i], y_pred[:, i]) for i in range(y_test_np.shape[1])]

    return input_data, output_data, mse, mse_columns, y_test_np, y_pred, total_loss

def load_and_predict(input_data_path, output_data_path, model_save_path='models.pth'):
    input_header, input_data = read_txt_file(input_data_path)
    output_header, output_data = read_txt_file(output_data_path)

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(input_data)

    # 转换为PyTorch张量
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(output_data, dtype=torch.float32)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.3, random_state=42)

    # 创建DataLoaders
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # 实例化特征提取模型并加载预训练模型
    feature_extractor = CNN1DFeatureExtractor(X_train.shape[1])
    feature_extractor, xgb_models = load_models(feature_extractor, model_save_path)
    print(f'Models loaded from {model_save_path}')

    # 提取训练集和测试集的特征
    X_train_features, y_train_targets = extract_features(train_loader, feature_extractor)
    X_test_features, y_test_targets = extract_features(test_loader, feature_extractor)

    # 将提取的特征转换为numpy数组
    X_train_np = X_train_features.numpy()
    y_train_np = y_train_targets.numpy()
    X_test_np = X_test_features.numpy()
    y_test_np = y_test_targets.numpy()

    # 使用XGBoost进行回归
    y_pred = np.column_stack([model.predict(X_test_np) for model in xgb_models])

    mse = mean_squared_error(y_test_np, y_pred)
    mse_columns = [mean_squared_error(y_test_np[:, i], y_pred[:, i]) for i in range(y_test_np.shape[1])]

    return input_data, output_data, mse, mse_columns, y_test_np, y_pred

def predict_new_data(new_data_path, model_save_path='models.pth'):
    _, new_data = read_txt_file(new_data_path)

    scaler = StandardScaler()
    new_data_scaled = scaler.fit_transform(new_data)

    new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

    feature_extractor = CNN1DFeatureExtractor(new_data_tensor.shape[1])
    feature_extractor, xgb_models = load_models(feature_extractor, model_save_path)
    print(f'Models loaded from {model_save_path}')

    feature_extractor.eval()
    with torch.no_grad():
        new_features = feature_extractor(new_data_tensor).numpy()

    predictions = np.column_stack([model.predict(new_features) for model in xgb_models])

    return predictions

def plot_loss_curve(total_loss):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure()
    plt.plot(total_loss, label='训练损失')
    plt.xlabel('迭代次数')
    plt.ylabel('Loss')
    plt.title('总训练损失曲线')
    plt.legend()
    plt.savefig('LossCurve.jpg')
    plt.close()

def plot_predictions(test_y, ypred, mse_columns, output_data_path, start_idx=0, end_idx=100):
    output_header, output_data = read_txt_file(output_data_path)
    labels = output_header
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    test_y_partial = test_y[start_idx:end_idx]
    ypred_partial = ypred[start_idx:end_idx]

    for i in range(test_y_partial.shape[1]):
        plt.figure()
        plt.plot(test_y_partial[:, i].flatten(), marker='o', label=f'{labels[i]} 真实值')
        plt.plot(ypred_partial[:, i].flatten(), marker='^', label=f'{labels[i]} 预测值')
        plt.title(f'{labels[i]} 真实值与预测值对比\nMSE: {mse_columns[i]:.4f}')
        plt.legend()
        plt.show()
        plt.close()
