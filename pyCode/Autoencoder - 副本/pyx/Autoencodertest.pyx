import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


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


class Autoencoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            # nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(model, train_loader, criterion, optimizer, num_epochs):
    losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}')
    return losses


def extract_features(loader, model):
    model.eval()
    features = []
    targets = []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            features.append(outputs)
            targets.append(labels)
    return torch.cat(features), torch.cat(targets)


def train_and_predict(input_data_path, output_data_path):
    input_header, input_data = read_txt_file(input_data_path)
    output_header, output_data = read_txt_file(output_data_path)

    # scaler = StandardScaler()
    scaler_input = StandardScaler()
    scaler_output = StandardScaler()
    # input_data_scaled = scaler.fit_transform(input_data)
    # output_data_scaled = scaler.fit_transform(output_data)
    input_data_scaled = scaler_input.fit_transform(input_data)
    output_data_scaled = scaler_output.fit_transform(output_data)

    X_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)
    # y_tensor = torch.tensor(output_data_scaled, dtype=torch.float32)
    # y_tensor = torch.tensor(output_data, dtype=torch.float32)
    y_tensor = torch.tensor(output_data_scaled, dtype=torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.3, random_state=42)

    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    model = Autoencoder(X_train.shape[1], y_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses = train_autoencoder(model, train_loader, criterion, optimizer, num_epochs=50)

    # 使用整个模型进行预测而不是编码器部分
    model.eval()
    y_test_real = []
    y_pred_real = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)

            outputs = torch.tensor(scaler_output.inverse_transform(outputs.numpy()), dtype=torch.float32)
            labels = torch.tensor(scaler_output.inverse_transform(labels.numpy()), dtype=torch.float32)

            y_test_real.append(labels)
            y_pred_real.append(outputs)

    y_test_real = torch.cat(y_test_real).numpy()
    y_pred_real = torch.cat(y_pred_real).numpy()

    mse = mean_squared_error(y_test_real, y_pred_real)
    mse_columns = [mean_squared_error(y_test_real[:, i], y_pred_real[:, i]) for i in range(y_test_real.shape[1])]

    return input_data, output_data, mse, mse_columns, losses, y_test_real, y_pred_real


def plot_loss_curve(evals_result):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure()
    plt.plot(evals_result, label='训练损失')
    plt.xlabel('迭代次数')
    plt.ylabel('Loss')
    plt.title('训练损失曲线')
    plt.legend()
    plt.savefig('LossCurve.jpg')
    plt.close()


def plot_predictions(test_y, ypred, mse_columns, output_data_path, start_idx=0, end_idx=100):
    labels, _ = read_txt_file(output_data_path)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    test_y_partial = test_y[start_idx:end_idx]
    ypred_partial = ypred[start_idx:end_idx]
    print(f'test_y_partial.shape: {test_y_partial.shape}')
    print(f'ypred_partial.shape: {ypred_partial.shape}')

    # num_labels = min(len(labels), test_y_partial.shape[1])
    num_labels = test_y_partial.shape[1]
    print(f'num_labels: {num_labels}')
    if len(labels) < num_labels:
        labels = [f'输出{i + 1}' for i in range(num_labels)]

    for i in range(num_labels):
        plt.figure()
        plt.plot(test_y_partial[:, i].flatten(), marker='o', label=f'{labels[i]} 真实值')
        plt.plot(ypred_partial[:, i].flatten(), marker='^', label=f'{labels[i]} 预测值')
        plt.title(f'{labels[i]} 真实值与预测值对比\nMSE: {mse_columns[i]:.4f}')
        plt.legend()
        plt.show()
        plt.close()
