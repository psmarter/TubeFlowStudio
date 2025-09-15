import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import torch
import sys
import chardet

def detect_file_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def read_txt_file(file_path):
    encoding = detect_file_encoding(file_path)
    with open(file_path, 'r', encoding=encoding) as file:
        lines = file.readlines()
    header = lines[0].strip().split()
    data = [list(map(float, line.strip().split())) for line in lines[1:]]
    return header, np.array(data)

# def read_txt_file(file_path, encoding='utf-8'):
#     with open(file_path, 'r', encoding=encoding) as file:
#         lines = file.readlines()
#
#     # 第一行为表头
#     header = lines[0].strip().split()
#
#     # 跳过第一行
#     data_lines = lines[1:]
#
#     data = []
#     for line in data_lines:
#         numbers = [float(num) for num in line.split() if num]
#         data.append(numbers)
#
#     return header, np.array(data)

def train_and_predict(input_data_path, output_data_path, model_save_path='kriging_model.pth'):
    input_header, input_data = read_txt_file(input_data_path)
    output_header, output_data = read_txt_file(output_data_path)

    scaler_input_data = StandardScaler()
    input_data_scaled = scaler_input_data.fit_transform(input_data)
    scaler_output_data = StandardScaler()
    output_data_scaled = scaler_output_data.fit_transform(output_data)

    input_data_train, input_data_test, output_data_train, output_data_test = train_test_split(input_data_scaled, output_data_scaled, test_size=0.3, random_state=42)
    kernel = C(1.0, (1e-4, 1e1)) * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e1)) + WhiteKernel(
        noise_level=1e-3)
    gpr = MultiOutputRegressor(
        GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2, normalize_y=True))
    gpr.fit(input_data_train, output_data_train)

    # 分阶段训练并记录损失
    num_epochs = 10
    evals_result = []

    for epoch in range(num_epochs):
        gpr.fit(input_data_train, output_data_train)
        output_data_pred_train = np.column_stack([estimator.predict(input_data_train) for estimator in gpr.estimators_])
        total_loss = mean_squared_error(output_data_train, output_data_pred_train)
        evals_result.append(total_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss}')
    output_data_pred = np.column_stack([estimator.predict(input_data_test) for estimator in gpr.estimators_])
    output_data_pred_real = scaler_output_data.inverse_transform(output_data_pred)
    output_data_test_real = scaler_output_data.inverse_transform(output_data_test)

    # 计算总的 MSE
    total_mse = mean_squared_error(output_data_test_real, output_data_pred_real)

    # 计算每列数据单独的 MSE
    mse_per_column = [mean_squared_error(output_data_test_real[:, i], output_data_pred_real[:, i]) for i in range(output_data_test_real.shape[1])]

    # 保存模型
    save_model(gpr, scaler_input_data, scaler_output_data, model_save_path)

    return total_mse, mse_per_column, output_data_test_real, output_data_pred_real, evals_result, input_data, output_data

def save_model(model, scaler_input, scaler_output, file_path):
    model_dict = {
        'model_state': model,
        'scaler_input': scaler_input,
        'scaler_output': scaler_output
    }
    torch.save(model_dict, file_path, _use_new_zipfile_serialization=False)

def load_model(file_path):
    model_dict = torch.load(file_path)
    model = model_dict['model_state']
    scaler_input = model_dict['scaler_input']
    scaler_output = model_dict['scaler_output']
    return model, scaler_input, scaler_output

def predict_new_data(new_data_path, model_save_path='kriging_model.pth'):
    _, new_data = read_txt_file(new_data_path)

    model, scaler_input, scaler_output = load_model(model_save_path)
    print(f'Model loaded from {model_save_path}')

    new_data_scaled = scaler_input.transform(new_data)
    new_data_pred_scaled = np.column_stack([estimator.predict(new_data_scaled) for estimator in model.estimators_])
    new_data_pred = scaler_output.inverse_transform(new_data_pred_scaled)

    return new_data_pred

def plot_predictions(test_y, ypred, mse_per_column, output_data_path, start_idx=0, end_idx=100):
    labels, _ = read_txt_file(output_data_path)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    test_y_partial = test_y[start_idx:end_idx]
    ypred_partial = ypred[start_idx:end_idx]

    for i in range(test_y_partial.shape[1]):
        plt.figure()
        plt.plot(test_y_partial[:, i].flatten(), marker='o', label=f'{labels[i]} 真实值')
        plt.plot(ypred_partial[:, i].flatten(), marker='^', label=f'{labels[i]} 预测值')
        plt.title(f'{labels[i]} 真实值与预测值对比\nMSE: {mse_per_column[i]:.4f}')
        plt.legend()
        plt.show()
        plt.close()

def plot_loss_curve(evals_result, input_data_path, output_data_path):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure()
    plt.plot(evals_result, label='训练损失')
    plt.xlabel('迭代次数')
    plt.ylabel('RMSE')
    plt.title('总训练损失曲线')
    plt.legend()
    plt.savefig('LossCurve.jpg')
    plt.close()
