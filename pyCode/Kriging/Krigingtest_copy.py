import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def train_and_predict(file_path):
    data = pd.read_excel(file_path)
    X = data.iloc[:, 2:-3].values  # 输入数据为从第三列开始到倒数第三列
    y = data.iloc[:, -3:].values  # 输出数据为最后三列

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)
    kernel = C(1.0, (1e-4, 1e1)) * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e1)) + WhiteKernel(
        noise_level=1e-3)
    gpr = MultiOutputRegressor(
        GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2, normalize_y=True))
    gpr.fit(X_train, y_train)

    y_pred = np.column_stack([estimator.predict(X_test) for estimator in gpr.estimators_])
    y_pred_real = scaler_y.inverse_transform(y_pred)
    y_test_real = scaler_y.inverse_transform(y_test)

    # 计算总的 MSE
    total_mse = mean_squared_error(y_test_real, y_pred_real)

    # 计算每列数据单独的 MSE
    mse_per_column = [mean_squared_error(y_test_real[:, i], y_pred_real[:, i]) for i in range(y_test_real.shape[1])]

    # 绘制损失曲线
    evals_result = gpr.estimators_[0].log_marginal_likelihood_value_
    #plot_loss_curve(evals_result)

    return total_mse, mse_per_column, y_test_real, y_pred_real, evals_result, X, y

def plot_loss_curve(evals_result):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    labels = ['归一化相速', '耦合阻抗', '衰减常数']
    plt.figure()
    for i, result in enumerate(evals_result):
        plt.plot(result['validation_0']['rmse'], label=f'{labels[i]} RMSE')
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    plt.savefig('LossCurve.jpg')
    plt.close()