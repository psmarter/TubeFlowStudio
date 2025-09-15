import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch


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

def train_and_predict(input_data_path, output_data_path, model_save_path='models.pth'):
    input_header, input_data = read_txt_file(input_data_path)
    output_header, output_data = read_txt_file(output_data_path)

    P_array = np.array(input_data)
    if len(output_header) == 1:
        C_array = np.array(output_data).reshape(-1, 1)
    else:
        C_array = np.array(output_data)
    Paras_all = P_array  # N 个尺寸的结果 (N,4)
    Curves_all = C_array  # N 个曲线的结果  (N,3,1)
    X = Paras_all
    y = Curves_all

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=4)
    other_params = {
        'eta': 0.1,  # 学习率
        'reg_alpha': 0.01,  # L1 正则化参数
        'reg_lambda': 0.01,  # L2 正则化参数
        'max_depth': 6  # 树的最大深度
    }
    models = []
    evals_result = []
    for i in range(train_y.shape[1]):
        model = xgb.XGBRegressor(objective='reg:squarederror', **other_params)
        model.fit(train_X, train_y[:, i], eval_set=[(test_X, test_y[:, i])], eval_metric='rmse', verbose=True)
        models.append(model)
        evals_result.append(model.evals_result())

    ypred = np.column_stack([model.predict(test_X) for model in models])
    mse = mean_squared_error(test_y, ypred)
    # 计算每列数据的单独 MSE
    mse_columns = [mean_squared_error(test_y[:, i], ypred[:, i]) for i in range(test_y.shape[1])]
    # 保存模型
    save_models(models, model_save_path)
    return test_y, ypred, mse, mse_columns, models, evals_result, input_data, output_data

def save_models(models, file_path):
    models_dict = {
        'xgb_models': [model.get_booster().save_raw() for model in models]
    }
    torch.save(models_dict, file_path)

def load_models(file_path):
    models_dict = torch.load(file_path)
    xgb_models = []
    for model_str in models_dict['xgb_models']:
        model = xgb.XGBRegressor(objective='reg:squarederror')
        model.load_model(model_str)
        xgb_models.append(model)
    return xgb_models

def predict_new_data(new_data_path, model_save_path='models.pth'):
    _, new_data = read_txt_file(new_data_path)

    new_data_np = np.array(new_data)

    xgb_models = load_models(model_save_path)
    print(f'Models loaded from {model_save_path}')

    ypred = np.column_stack([model.predict(new_data_np) for model in xgb_models])

    return ypred

def plot_predictions(test_y, ypred, mse_columns, output_data_path, start_idx=0, end_idx=100):
    labels, _ = read_txt_file(output_data_path)
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

def plot_loss_curve(evals_result, input_data_path, output_data_path):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    output_header, output_data = read_txt_file(output_data_path)
    labels = output_header
    plt.figure()
    for i, result in enumerate(evals_result):
        plt.plot(result['validation_0']['rmse'], label=f'{labels[i]}')
    plt.title('总损失曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('LossCurve.jpg')
    plt.close()


