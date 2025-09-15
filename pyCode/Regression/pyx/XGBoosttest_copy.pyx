import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def train_and_predict(file_path):
    input_table = pd.read_excel(file_path)  # 读取表格
    input_table_rows = input_table.shape[0]  # 行数
    input_table_colunms = input_table.shape[1]  # 列数
    input_table_header = input_table.columns.values.tolist()  # 表头
    xy = np.array(input_table)
    x_data = xy[:, 2:-3]  # 第三列到倒数第二列
    y_data = xy[:, -3:]  # 倒数第三列到最后一列
    P_array = np.array(x_data)
    C_array = np.array(y_data)
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

    return test_y, ypred, mse, models, evals_result, x_data, y_data

def plot_loss_curve(evals_result):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    labels = ['归一化相速', '耦合阻抗', '衰减常数']
    plt.figure()
    for i, result in enumerate(evals_result):
        plt.plot(result['validation_0']['rmse'], label=f'{labels[i]} RMSE')
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('LossCurve.jpg')
    plt.close()


